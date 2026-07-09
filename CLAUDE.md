# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch CLIP-style image–text retrieval model, trained to align image embeddings with caption embeddings on Flickr8k / Flickr30k. The pipeline is written in plain PyTorch (the training loop is hand-rolled, not PyTorch Lightning — see the caveat below). Two encoders (a `timm` vision backbone and a HuggingFace text model) are each followed by a projection head into a shared embedding space, trained with a distillation-style contrastive loss.

## Environment

Uses a conda environment (see `.vscode/settings.json`). Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` is now a full pinned freeze (includes `torch`, `timm`, `transformers`, `albumentations`, `opencv-python-headless`, `pandas`, `wandb`, `lightning`). Note it pins `opencv-python-headless` (not the GUI `opencv-python`).

## Running training

Training is a standalone script under `src/trainers/`. It adds `src/` to `sys.path`, so run it directly:

```bash
python src/trainers/trainer.py --dataset-name flickr8k --dataset-path datasets/flickr8k
```

Key flags (see `main()` in `src/trainers/trainer.py` for the full list): `--dataset-name` (`flickr8k`/`flickr30k`), `--dataset-path` (local dir), `--epochs`, `--batch-size`, `--size` (image resize), `--max-length` (caption tokens), and three separate learning rates: `--image-encoder-lr`, `--text-encoder-lr`, `--head-lr`. The best checkpoint (lowest val loss) is saved via `torch.save(state_dict)` to `--output` (default `best.pt`).

The image/text encoder **aliases are hardcoded** at the top of `trainer.py` (`IMAGE_ENCODER_ALIAS = "resnet50"`, `TEXT_ENCODER_ALIAS = "distilbert-base-uncased"`), not exposed as CLI flags. Change them there.

There is no test suite, linter config, or build step in the repo.

## Architecture

### Data loading (local disk — NOT W&B)

Datasets are read from **local directories** passed as `dataset_path`. (An earlier design fetched data from Weights & Biases artifacts; that is gone — `wandb` remains in requirements but the training path does not use it.)

- `src/dataloaders/ImageRetrievalDataset.py` — `ImageRetrievalDataset`, abstract base `Dataset`. `__init__` calls the subclass hook `fetch_dataset()`, tokenizes **all** captions up front with a HuggingFace tokenizer (`padding=True, truncation=True`, `return_tensors='pt'`), and builds Albumentations resize+normalize transforms. `__getitem__` returns a dict: tokenizer outputs (`input_ids`, `attention_mask`, ...) sliced per-index, plus `image` (CHW float tensor loaded via `cv2`, BGR→RGB) and `caption` (raw string). Subclasses implement `fetch_dataset()` → `(image_files, captions)`.
- `src/dataloaders/flickr.py` — `Flickr8kDataset` / `Flickr30kDataset`. Each reads local annotation files and `assert`s every image path exists. Flickr8k: `captions.txt` (comma-separated, columns `image`/`caption`, images under `Images/`). Flickr30k: `results.csv` (`|`-separated, columns `image_name`/` comment` — note the leading space; `dropna()` applied; images under `flickr30k_images/`).

### Model (`src/models/`)

- `src/models/clip_model.py` — `CLIPModel` (a plain `nn.Module`). `forward(batch)` runs both encoders, projects each into `projection_dims`, and **returns the scalar loss directly** (not embeddings). Encoder aliases and dims are constructor args (`image_embedding_dims=2048` for resnet50, `text_embedding_dims=768` for distilbert, `projection_dims=256`).
- `src/models/encoders/image_encoders.py` — `ImageEncoder` wraps `timm.create_model(..., num_classes=0, global_pool="avg")` to get a pooled feature vector.
- `src/models/encoders/text_encoders.py` — `TextEncoder` wraps `transformers.AutoModel` and takes the **CLS token** (`last_hidden_state[:, 0, :]`) as the sentence embedding.
- `src/models/embeddings_connectors/embeddings_conector.py` (filename misspelled) — three projection heads: `LinearEmbeddingsConnector`, `MLPEmbeddingsConnector` (Linear→GELU→Linear→dropout + residual + LayerNorm), and `SwiGLUEmbeddingsConnector`. **`CLIPModel` currently hardcodes `MLPEmbeddingsConnector`** for both heads; the other two are available but unwired.

### Loss

`CLIPModel._compute_losses` is the distillation-style CLIP loss (à la the Moein Shariatnia / Keras CLIP tutorial), **not** the symmetric InfoNCE from the original paper: soft targets are `softmax((image_sim + text_sim)/2 * temperature)`, and image/text cross-entropy losses are averaged. Note: embeddings are **not L2-normalized** before the dot products, and `temperature` is *multiplied* into the targets but *divides* the logits — deliberate quirks of this formulation, so preserve them unless intentionally changing the loss.

## Known dead / orphaned code

These exist but are **not** on the training path — reconcile before wiring them in:

- `src/dataloaders/ImageRetrievalDataModule.py` — a `LightningDataModule`. Fully functional but **unused**: `trainer.py` builds its own `DataLoader`s (`build_loaders`) and has its own copy of `DATASET_LOOKUP`. Only reach for the DataModule if reintroducing a Lightning training path.
- `src/dataloaders/utils.py` — `collate_fn` expects **tuple-shaped** items `(img, caption, input_id, attention_mask)`, but `__getitem__` returns a **dict**. It is imported by the DataModule but never passed to a `DataLoader`; the default collate is used everywhere. It also has a leftover `print(time.time() - start)`.
- `logs/` — CSV metrics + `hparams.yaml` from earlier PyTorch-Lightning `CSVLogger` runs. The current `trainer.py` does not write here.

## Conventions

- New datasets: subclass `ImageRetrievalDataset`, implement `fetch_dataset()`, and register in **both** `DATASET_LOOKUP` dicts (`trainer.py` and `ImageRetrievalDataModule.py`) if you want it usable from either.
- New projection heads go in `embeddings_conector.py`; to use one, swap the class in `CLIPModel.__init__`.
- `datasets/` (local data) and `notebooks/` are gitignored experimental scratch space, not part of the package.
