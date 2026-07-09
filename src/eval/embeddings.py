"""Extracción de embeddings de un subconjunto, listos para métricas.

`CLIPModel.forward` devuelve la loss, no embeddings: aquí se usan `encode_image` /
`encode_text` directamente.
"""

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class SubsetEmbeddings:
    """`gt[j]` es el índice en `gallery_paths` de la imagen correcta de la caption `j`.

    `image_embeddings` va indexado por galería (imágenes únicas) y `text_embeddings`
    por caption, así que sus longitudes difieren: hay ~5 captions por imagen.
    """

    image_embeddings: torch.Tensor  # [G, d]
    text_embeddings: torch.Tensor  # [N, d]
    gt: torch.Tensor  # [N], valores en [0, G)
    gallery_paths: List[str]


class _GalleryImages(Dataset):
    """Solo imágenes, sin captions: cada imagen de la galería se codifica una vez.

    Iterar el dataset original la decodificaría una vez por caption (~5x).
    """

    def __init__(self, paths: Sequence[str], transforms) -> None:
        self.paths = list(paths)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        return torch.tensor(image).permute(2, 0, 1).float()


@torch.no_grad()
def encode_subset(
    model,
    dataset,
    indices: Sequence[int],
    device,
    batch_size: int = 64,
    num_workers: int = 2,
) -> SubsetEmbeddings:
    model.eval()

    gallery_paths: List[str] = []
    gallery_index_of = {}
    gt = []
    for caption_index in indices:
        path = dataset.image_files[caption_index]
        if path not in gallery_index_of:
            gallery_index_of[path] = len(gallery_paths)
            gallery_paths.append(path)
        gt.append(gallery_index_of[path])

    image_loader = DataLoader(
        _GalleryImages(gallery_paths, dataset.transforms),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    image_embeddings = [
        model.encode_image(images.to(device)).cpu()
        for images in tqdm(image_loader, desc="imágenes")
    ]

    # Las captions ya están tokenizadas en bloque por el dataset; recorrer el
    # DataLoader solo para leerlas volvería a decodificar las imágenes.
    caption_indices = torch.as_tensor(list(indices))
    input_ids = dataset.tokenized_captions["input_ids"][caption_indices]
    attention_mask = dataset.tokenized_captions["attention_mask"][caption_indices]

    text_embeddings = []
    for start in tqdm(range(0, len(caption_indices), batch_size), desc="captions"):
        chunk = slice(start, start + batch_size)
        text_embeddings.append(
            model.encode_text(
                input_ids=input_ids[chunk].to(device),
                attention_mask=attention_mask[chunk].to(device),
            ).cpu()
        )

    return SubsetEmbeddings(
        image_embeddings=torch.cat(image_embeddings).float(),
        text_embeddings=torch.cat(text_embeddings).float(),
        gt=torch.as_tensor(gt),
        gallery_paths=gallery_paths,
    )
