"""Recall@K, rank y MRR en las dos direcciones (texto→imagen e imagen→texto)."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def _rank_metrics(ranks: torch.Tensor, ks: Sequence[int]) -> Dict[str, float]:
    """`ranks` son rangos base 0 del acierto; se reportan base 1."""
    ranks = ranks.float()
    metrics = {f"recall@{k}": (ranks < k).float().mean().item() * 100 for k in ks}
    # R@K satura y no dice cuán lejos quedan los fallos; la mediana sí.
    metrics["median_rank"] = (ranks + 1).median().item()
    metrics["mean_rank"] = (ranks + 1).mean().item()
    metrics["mrr"] = (1.0 / (ranks + 1)).mean().item()
    return metrics


def retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    gt: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Dict[str, float]]:
    """Rankea por similitud coseno. `gt[j]` = índice de galería de la caption `j`.

    La loss de entrenamiento no normaliza L2, pero para rankear el coseno es la
    medida correcta: si no, una imagen con embedding de norma grande sube en el
    ranking de todas las consultas.
    """
    images = F.normalize(image_embeddings, p=2, dim=-1)
    texts = F.normalize(text_embeddings, p=2, dim=-1)

    similarity = texts @ images.T  # [N captions, G imágenes]
    n_captions, n_gallery = similarity.shape
    captions = torch.arange(n_captions)

    # Los empates se resuelven en contra del acierto (cuentan como candidatos por
    # delante). Sin esto un modelo colapsado, cuyas similitudes son todas idénticas,
    # sacaría Recall@1 del 100%: ningún distractor puntúa *estrictamente* más.
    positive = similarity[captions, gt].unsqueeze(1)
    t2i_ranks = (similarity >= positive).sum(dim=1) - 1

    # imagen→texto: acierta si *cualquiera* de las ~5 captions de la imagen entra en
    # el top-K (protocolo estándar de Flickr), luego el rango lo fija su mejor caption.
    best_positive = torch.full((n_gallery,), float("-inf"))
    best_positive.scatter_reduce_(0, gt, similarity[captions, gt], reduce="amax")
    best_positive = best_positive.unsqueeze(0)

    is_positive = torch.zeros_like(similarity, dtype=torch.bool)
    is_positive[captions, gt] = True
    ahead = similarity > best_positive
    tied = (similarity == best_positive) & ~is_positive
    i2t_ranks = (ahead | tied).sum(dim=0)

    return {
        "text_to_image": _rank_metrics(t2i_ranks, ks),
        "image_to_text": _rank_metrics(i2t_ranks, ks),
        "gallery_size": n_gallery,
        "n_captions": n_captions,
    }
