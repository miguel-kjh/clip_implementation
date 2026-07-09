"""Diagnóstico geométrico de los embeddings: ¿se ha colapsado el espacio?

La val loss de este proyecto no lo detecta. `CLIPModel._compute_losses` construye
los targets a partir de los propios embeddings, así que "todos los embeddings son
iguales" la satisface trivialmente en ln(batch_size). Estas métricas miran la
geometría directamente.
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F

_EPS = 1e-12
_UNIFORMITY_SAMPLE = 4096


def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.double())


def effective_rank(normalized: torch.Tensor) -> float:
    """RankMe: exp(entropía del espectro de valores singulares).

    Es la métrica que caza el colapso *dimensional parcial* —el espacio usa 12 de
    256 dimensiones— que el coseno medio no ve, porque los embeddings pueden estar
    bien repartidos dentro de un subespacio diminuto.
    """
    singular_values = _singular_values(normalized)
    p = singular_values / (singular_values.sum() + _EPS) + _EPS
    return math.exp(float(-(p * p.log()).sum()))


def mean_offdiagonal_cosine(normalized: torch.Tensor) -> float:
    """Coseno medio entre pares distintos. ~1.0 = colapso total.

    La suma de todos los productos escalares por pares es ‖Σᵢ eᵢ‖², luego el
    off-diagonal sale en O(N·d) sin materializar la matriz N×N.
    """
    n = normalized.shape[0]
    if n < 2:
        return float("nan")
    total = normalized.double().sum(dim=0).pow(2).sum()
    return float((total - n) / (n * (n - 1)))


def uniformity(normalized: torch.Tensor, seed: int = 0) -> float:
    """log E exp(-2‖u-v‖²) (Wang & Isola 2020). Más negativo = mejor repartido."""
    n = normalized.shape[0]
    if n < 2:
        return float("nan")
    if n > _UNIFORMITY_SAMPLE:
        generator = torch.Generator().manual_seed(seed)
        sample = torch.randperm(n, generator=generator)[:_UNIFORMITY_SAMPLE]
        normalized = normalized[sample]

    square_distances = torch.pdist(normalized.double(), p=2).pow(2)
    return float(square_distances.mul(-2).exp().mean().log())


def collapse_metrics(embeddings: torch.Tensor, seed: int = 0) -> Dict[str, float]:
    """Diagnóstico de una modalidad. `embeddings` sin normalizar, [N, d]."""
    normalized = F.normalize(embeddings, p=2, dim=-1)
    n, dims = embeddings.shape

    centered = normalized.double() - normalized.double().mean(dim=0, keepdim=True)
    variances = _singular_values(centered).pow(2)
    variance_ratio = variances / (variances.sum() + _EPS)

    rank = effective_rank(normalized)
    norms = embeddings.norm(dim=-1)
    return {
        # Ojo: se queda alto (~0.9) incluso en modelos sanos, por el "cone effect" de
        # CLIP (todos los embeddings comparten una componente media grande). Solo es
        # señal de colapso pasado ~0.99; el discriminador fiable es `effective_rank`,
        # que vale ~1 si colapsa y ~100 con un cono sano.
        "mean_cosine": mean_offdiagonal_cosine(normalized),
        "effective_rank": rank,
        "effective_rank_ratio": rank / dims,
        "pca_var_top1": float(variance_ratio[0]),
        "pca_var_top10": float(variance_ratio[:10].sum()),
        # Sano ≈ 1/sqrt(d) si la masa se reparte por todas las dimensiones.
        "mean_per_dim_std": float(normalized.std(dim=0).mean()),
        # Sobre los embeddings SIN normalizar: caza la divergencia de norma de la
        # cabeza `swiglu`, la única que no termina en LayerNorm.
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()) if n > 1 else float("nan"),
        "uniformity": uniformity(normalized, seed=seed),
    }


def alignment(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, gt: torch.Tensor
) -> float:
    """E‖e_img - e_txt‖² sobre los pares correctos. Más bajo = positivos más juntos.

    Junto con `uniformity` separa las dos mitades que la loss mezcla: acercar
    positivos y repartir el espacio. Un modelo colapsado tiene alignment excelente
    y uniformity pésima.
    """
    images = F.normalize(image_embeddings, p=2, dim=-1)
    texts = F.normalize(text_embeddings, p=2, dim=-1)
    return float((texts - images[gt]).pow(2).sum(dim=-1).mean())


def modality_gap(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
) -> float:
    """Distancia entre los centroides de cada modalidad.

    Importa aquí más de lo normal porque la loss no normaliza L2: las dos ramas
    pueden acabar en conos separados aunque cada una internamente esté sana.
    """
    images = F.normalize(image_embeddings, p=2, dim=-1).mean(dim=0)
    texts = F.normalize(text_embeddings, p=2, dim=-1).mean(dim=0)
    return float((images - texts).norm())
