"""Evalúa un experimento entrenado: calidad de retrieval y colapso de embeddings.

    python src/eval/evaluate.py --experiment-dir checkpoints/<exp>

Escribe `metrics.json` junto al `hparams.json` del experimento.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.checkpoint import load_clip_model, load_hparams
from models.clip_model import CLIPModel
from trainers.trainer import DATASET_LOOKUP, seed_everything

from eval.collapse import alignment, collapse_metrics, modality_gap
from eval.embeddings import encode_subset
from eval.retrieval import retrieval_metrics
from eval.splits import (
    image_disjoint_indices,
    reproduce_val_indices,
    subsample_gallery,
)

SPLITS = ("val", "val-disjoint")

# Umbrales de la interpretación impresa.
# Un modelo sano de este repo da coseno medio ~0.91 (cone effect) con rango efectivo
# ~40-70, así que el coseno solo delata colapso pasado 0.99 (CLAUDE.md: el run con
# --dropout 0 dio 0.9999). El rango efectivo es el criterio fiable. `norm_mean` alto
# es la divergencia de la cabeza `swiglu`.
COLLAPSE_COSINE = 0.99
COLLAPSE_RANK = 2.0
COLLAPSE_RANK_RATIO = 0.05
CONE_COSINE = 0.9
DIVERGING_NORM = 50.0


def build_dataset(hparams, tokenizer):
    data = hparams["data"]
    return DATASET_LOOKUP[data["dataset_name"]](
        dataset_path=data["dataset_path"],
        tokenizer=tokenizer,
        target_size=data["size"],
        max_length=data["max_length"],
    )


def evaluate_split(model, dataset, indices, device, args, seed):
    embeddings = encode_subset(
        model,
        dataset,
        indices,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    images, texts, gt = (
        embeddings.image_embeddings,
        embeddings.text_embeddings,
        embeddings.gt,
    )
    return {
        "retrieval": retrieval_metrics(images, texts, gt, ks=args.ks),
        "collapse": {
            "image": collapse_metrics(images, seed=seed),
            "text": collapse_metrics(texts, seed=seed),
        },
        "cross_modal": {
            "alignment": alignment(images, texts, gt),
            "modality_gap": modality_gap(images, texts),
        },
    }


def diagnose(metrics, projection_dims):
    """Frases de interpretación; lista vacía = nada anómalo."""
    warnings = []
    for modality, stats in metrics["collapse"].items():
        rank, cosine = stats["effective_rank"], stats["mean_cosine"]
        collapsed = rank < COLLAPSE_RANK or cosine > COLLAPSE_COSINE
        if collapsed:
            warnings.append(
                f"COLAPSO ({modality}): rango efectivo {rank:.2f}, coseno medio "
                f"{cosine:.4f}. Los embeddings apuntan casi todos al mismo sitio; "
                f"el Recall será el del azar. ¿Entrenaste con --dropout 0?"
            )
        elif stats["effective_rank_ratio"] < COLLAPSE_RANK_RATIO:
            warnings.append(
                f"COLAPSO DIMENSIONAL ({modality}): rango efectivo {rank:.1f} sobre "
                f"{projection_dims} dims. El espacio vive en un subespacio diminuto."
            )
        elif cosine > CONE_COSINE:
            warnings.append(
                f"cone effect ({modality}): coseno medio {cosine:.4f} pero rango "
                f"efectivo {rank:.1f}. Normal en CLIP, no es colapso: los embeddings "
                f"comparten una componente media grande y varían alrededor de ella."
            )
        if stats["norm_mean"] > DIVERGING_NORM:
            warnings.append(
                f"NORMA DIVERGENTE ({modality}): norma media {stats['norm_mean']:.1f}. "
                f"Típico de la cabeza `swiglu`, que no termina en LayerNorm."
            )
    return warnings


def format_report(name, metrics, projection_dims):
    lines = [f"\n{'=' * 72}", f"SPLIT: {name}", "=" * 72]

    retrieval = metrics["retrieval"]
    lines.append(
        f"{retrieval['n_captions']} captions sobre una galería de "
        f"{retrieval['gallery_size']} imágenes únicas "
        f"(azar: R@1 = {100 / retrieval['gallery_size']:.2f}%)"
    )
    lines.append("\n-- Retrieval " + "-" * 59)
    ks = sorted(
        int(key.split("@")[1])
        for key in retrieval["text_to_image"]
        if key.startswith("recall@")
    )
    labels = [f"R@{k}" for k in ks] + ["medRank", "MRR"]
    lines.append(f"{'dirección':<16}" + "".join(f"{label:>10}" for label in labels))
    for direction in ("text_to_image", "image_to_text"):
        stats = retrieval[direction]
        recalls = "".join(f"{stats[f'recall@{k}']:>9.2f}%" for k in ks)
        lines.append(
            f"{direction:<16}{recalls}{stats['median_rank']:>10.0f}"
            f"{stats['mrr']:>10.3f}"
        )

    lines.append("\n-- Colapso " + "-" * 61)
    columns = (
        ("cos medio", "mean_cosine", "{:>12.4f}"),
        ("rango efec", "effective_rank", "{:>12.1f}"),
        ("PCA top1", "pca_var_top1", "{:>12.3f}"),
        ("std/dim", "mean_per_dim_std", "{:>12.4f}"),
        ("norma", "norm_mean", "{:>12.2f}"),
        ("uniformity", "uniformity", "{:>12.3f}"),
    )
    lines.append(f"{'modalidad':<12}" + "".join(f"{label:>12}" for label, _, _ in columns))
    for modality, stats in metrics["collapse"].items():
        row = "".join(fmt.format(stats[key]) for _, key, fmt in columns)
        lines.append(f"{modality:<12}{row}")
    lines.append(f"(sano: std/dim ≈ {1 / projection_dims ** 0.5:.4f}, rango efectivo alto)")

    cross = metrics["cross_modal"]
    lines.append(
        f"\nalignment (pares correctos): {cross['alignment']:.4f}   "
        f"modality gap: {cross['modality_gap']:.4f}"
    )

    warnings = diagnose(metrics, projection_dims)
    lines.append("\n-- Diagnóstico " + "-" * 57)
    lines.extend(f"  ! {w}" for w in warnings)
    if not warnings:
        lines.append("  Sin señales de colapso.")
    return "\n".join(lines)


def print_comparability_warnings(results):
    """Avisa de los dos confounds que invalidan comparar el Recall entre splits."""
    if len(results) < 2:
        return
    retrievals = [r["retrieval"] for r in results.values()]

    gallery_sizes = {r["gallery_size"] for r in retrievals}
    if len(gallery_sizes) > 1:
        print(
            f"\n! Galerías de distinto tamaño {sorted(gallery_sizes)}: el Recall@K NO "
            f"es comparable entre splits, porque cuantas más imágenes compiten más "
            f"difícil es acertar. Relanza con --gallery-size para igualarlas."
        )

    densities = {round(r["n_captions"] / r["gallery_size"], 1) for r in retrievals}
    if len(densities) > 1:
        print(
            f"\n! Distinto número de captions por imagen {sorted(densities)}: el "
            f"Recall de imagen→texto NO es comparable entre splits, porque acierta si "
            f"*cualquiera* de las captions de la imagen entra en el top-K y unos "
            f"splits tienen más candidatas correctas que otros. El split a nivel de "
            f"caption reparte las ~5 captions de cada imagen entre train y val; el "
            f"disjunto se las queda todas. Texto→imagen sí es comparable."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=SPLITS,
        help="Repetible. Por defecto, ambos.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--gallery-size",
        type=int,
        default=None,
        help="Recorta cada split a N imágenes. Necesario para comparar el R@K entre "
        "splits: sin esto sus galerías tienen tamaños distintos y las cifras no se "
        "pueden comparar. El protocolo estándar de Flickr usa 1000.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Evalúa el modelo sin cargar los pesos, como control de azar.",
    )
    args = parser.parse_args()
    splits = args.splits or list(SPLITS)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.untrained:
        hparams = load_hparams(args.experiment_dir)
        model_hparams = dict(hparams["model"])
        model_hparams["image_encoder_pretrained"] = False
        model = CLIPModel(**model_hparams).eval()
        print("Control de azar: modelo SIN entrenar (pesos aleatorios).")
    else:
        model, hparams = load_clip_model(args.experiment_dir, map_location=device)
    model = model.to(device)

    seed = hparams["training"]["seed"]
    val_split = hparams["data"]["val_split"]
    projection_dims = hparams["model"]["projection_dims"]

    tokenizer = AutoTokenizer.from_pretrained(hparams["model"]["text_encoder_alias"])
    # Réplica del orden del trainer: sembrar, construir el dataset y solo entonces
    # partir. `reproduce_val_indices` depende de que nada consuma el RNG entre medias.
    seed_everything(seed)
    dataset = build_dataset(hparams, tokenizer)

    indices_by_split = {}
    if "val" in splits:
        indices_by_split["val"] = reproduce_val_indices(dataset, val_split)
        expected = int(val_split * len(dataset))
        assert len(indices_by_split["val"]) == expected, (
            f"El val split reproducido tiene {len(indices_by_split['val'])} captions, "
            f"se esperaban {expected}."
        )
    if "val-disjoint" in splits:
        indices_by_split["val-disjoint"] = image_disjoint_indices(
            dataset.image_files, val_split, seed
        )

    if args.gallery_size:
        indices_by_split = {
            name: subsample_gallery(
                dataset.image_files, indices, args.gallery_size, seed
            )
            for name, indices in indices_by_split.items()
        }

    results = {}
    for name, indices in indices_by_split.items():
        print(f"\nEvaluando split '{name}' ({len(indices)} captions)...")
        results[name] = evaluate_split(model, dataset, indices, device, args, seed)
        print(format_report(name, results[name], projection_dims))

    print_comparability_warnings(results)

    output_path = os.path.join(args.experiment_dir, args.output)
    if args.untrained:
        print("\nControl de azar: no se escribe metrics.json.")
    else:
        with open(output_path, "w") as f:
            json.dump({"experiment": args.experiment_dir, "splits": results}, f, indent=2)
        print(f"\nMétricas escritas en {output_path}")


if __name__ == "__main__":
    main()
