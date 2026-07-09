"""Subconjuntos de evaluación, en índices de caption.

El dataset es a nivel de caption (~5 por imagen), así que hay dos formas de
partirlo y no dan la misma cifra de Recall. Ver `image_disjoint_indices`.
"""

from typing import List, Sequence

import torch
from torch.utils.data import random_split


def reproduce_val_indices(dataset, val_split: float) -> List[int]:
    """Los índices de caption del val split sobre el que se eligió `best.pt`.

    `build_loaders` en `trainers/trainer.py` llama a `random_split` sin pasarle un
    generador, así que la partición sale del RNG global de torch.

    Precondición: hay que llamar a esta función replicando la secuencia del trainer
    —`seed_everything(seed)`, construir el dataset, y partir— sin que nada consuma
    el RNG de torch entre medias. Si esa invariante se rompe, la partición deja de
    coincidir en silencio.
    """
    val_length = int(val_split * len(dataset))
    train_length = len(dataset) - val_length
    _, val_subset = random_split(dataset, [train_length, val_length])
    return list(val_subset.indices)


def subsample_gallery(
    image_files: Sequence[str], indices: Sequence[int], gallery_size: int, seed: int
) -> List[int]:
    """Recorta el subconjunto a `gallery_size` imágenes (con todas sus captions).

    Recall@K depende muchísimo del tamaño de la galería: cuantas más imágenes
    compiten, más difícil acertar. Comparar el R@K de dos splits con galerías de
    distinto tamaño no mide nada. Igualarlas con esta función sí.
    """
    captions_by_image = {}
    for caption_index in indices:
        captions_by_image.setdefault(image_files[caption_index], []).append(caption_index)

    unique_images = list(captions_by_image)
    if gallery_size >= len(unique_images):
        return sorted(indices)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(unique_images), generator=generator).tolist()
    kept = [unique_images[i] for i in permutation[:gallery_size]]
    return sorted(i for image in kept for i in captions_by_image[image])


def image_disjoint_indices(
    image_files: Sequence[str], val_split: float, seed: int
) -> List[int]:
    """Índices de caption de un val split en el que ninguna imagen aparece en train.

    Se permutan las imágenes únicas y se aparta la fracción `val_split` de ellas;
    el subconjunto son todas las captions de esas imágenes. Es la cifra honesta:
    con el split a nivel de caption el modelo ya vio en entrenamiento casi toda
    imagen de validación (solo cambia la caption), y el Recall sale inflado.
    """
    captions_by_image = {}
    for caption_index, path in enumerate(image_files):
        captions_by_image.setdefault(path, []).append(caption_index)

    unique_images = list(captions_by_image)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(unique_images), generator=generator).tolist()

    n_val_images = int(val_split * len(unique_images))
    val_images = [unique_images[i] for i in permutation[:n_val_images]]

    return sorted(i for image in val_images for i in captions_by_image[image])
