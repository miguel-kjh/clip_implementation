"""Guardado y carga de experimentos: pesos + hiperparámetros de construcción."""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from .clip_model import CLIPModel

HPARAMS_FILENAME = "hparams.json"
WEIGHTS_FILENAME = "best.pt"


def save_hparams(experiment_dir: str, hparams: Dict[str, Any]) -> str:
    path = os.path.join(experiment_dir, HPARAMS_FILENAME)
    with open(path, "w") as f:
        json.dump(hparams, f, indent=2)
    return path


def load_hparams(experiment_dir: str) -> Dict[str, Any]:
    path = os.path.join(experiment_dir, HPARAMS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No hay '{HPARAMS_FILENAME}' en {experiment_dir!r}. Sin él no se puede "
            f"saber con qué conectores y dimensiones se construyó el modelo, así que "
            f"el state_dict no es reconstruible. Los experimentos anteriores a la "
            f"introducción de este fichero hay que reentrenarlos."
        )
    with open(path) as f:
        return json.load(f)


def _experiment_timestamp(path: str) -> float:
    """Fecha del experimento, del sufijo `_YYYYmmdd-HHMMSS` del nombre.

    Ordenar por nombre no sirve: el timestamp va al final, detrás de campos como
    `_ep1_` / `_ep4_`, que mandan en el orden alfabético.
    """
    match = re.search(r"_(\d{8}-\d{6})$", os.path.basename(path))
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S").timestamp()
    return os.path.getmtime(path)


def list_experiments(checkpoints_dir: str) -> List[Dict[str, Any]]:
    """Un registro por subcarpeta de `checkpoints_dir`, del más antiguo al más reciente.

    `hparams` es None y `error` explica el motivo cuando el experimento no es cargable.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"No existe el directorio {checkpoints_dir!r}")

    paths = [
        os.path.join(checkpoints_dir, name) for name in os.listdir(checkpoints_dir)
    ]
    paths = [p for p in paths if os.path.isdir(p)]

    experiments = []
    for path in sorted(paths, key=_experiment_timestamp):
        name = os.path.basename(path)

        record: Dict[str, Any] = {
            "name": name,
            "path": path,
            "hparams": None,
            "error": None,
        }
        try:
            record["hparams"] = load_hparams(path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            record["error"] = str(exc)

        weights = record["hparams"] or {}
        weights_filename = weights.get("training", {}).get("checkpoint", WEIGHTS_FILENAME)
        if not os.path.isfile(os.path.join(path, weights_filename)):
            record["error"] = record["error"] or f"Falta el fichero de pesos {weights_filename!r}"

        record["loadable"] = record["error"] is None
        experiments.append(record)
    return experiments


def load_clip_model(
    experiment_dir: str,
    weights_filename: Optional[str] = None,
    map_location: Any = "cpu",
):
    """Reconstruye el CLIPModel de un experimento y le carga sus pesos entrenados.

    Devuelve `(model.eval(), hparams)`. El modelo queda en `map_location`.
    """
    hparams = load_hparams(experiment_dir)
    if weights_filename is None:
        weights_filename = hparams.get("training", {}).get("checkpoint", WEIGHTS_FILENAME)

    model_hparams = dict(hparams["model"])
    # Los pesos de ImageNet los sobrescribe el state_dict entrenado; descargarlos
    # solo cuesta tiempo.
    model_hparams["image_encoder_pretrained"] = False

    model = CLIPModel(**model_hparams)
    state_dict = torch.load(
        os.path.join(experiment_dir, weights_filename),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.eval(), hparams
