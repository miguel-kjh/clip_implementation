# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch CLIP-style image–text retrieval implementation using PyTorch Lightning. The model (not yet committed) is trained to align image embeddings with caption embeddings on the Flickr8k / Flickr30k datasets. Currently only the data loading layer exists.

## Environment

Uses a conda environment (see `.vscode/settings.json`). Install with:

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` is incomplete — it omits several packages the code imports directly: `wandb`, `transformers`, `albumentations`, `opencv-python` (`cv2`), and `pandas`. Install these as well.

## Architecture

The data pipeline is built around Weights & Biases artifacts — datasets are not read from local disk paths but downloaded via the W&B API by `artifact_id`.

- `src/dataloaders/ImageRetrievalDataset.py` — `ImageRetrievalDataset`, an abstract base `Dataset`. In `__init__` it calls the subclass hook `fetch_dataset()`, tokenizes all captions up front with a HuggingFace tokenizer, and sets up Albumentations resize/normalize transforms. `__getitem__` returns a dict with tokenizer outputs (`input_ids`, `attention_mask`, ...) plus `image` (CHW float tensor via cv2) and `caption` (raw string). Subclasses must implement `fetch_dataset()` returning `(image_files, captions)`.
- `src/dataloaders/flickr.py` — `Flickr8kDataset` and `Flickr30kDataset`, concrete subclasses. Each downloads its W&B artifact and parses its annotation format: Flickr8k reads `captions.txt` (comma-separated, columns `image`/`caption`, images under `Images/`); Flickr30k reads `results.csv` (`|`-separated, columns `image_name`/` comment`, note the leading space, images under `flickr30k_images/`).
- `src/dataloaders/ImageRetrievalDataModule.py` — `ImageRetrievalDataModule` (LightningDataModule). Selects the dataset class via `DATASET_LOOKUP` (`"flickr8k"` / `"flickr30k"`), builds the tokenizer from `tokenizer_alias`, and in `setup()` random-splits into train/val by `val_split`.
- `src/dataloaders/utils.py` — `collate_fn`. **Stale/unused**: it expects tuple-shaped items `(img, caption, input_id, attention_mask)`, but the current `Dataset.__getitem__` returns a dict. The DataModule uses the default collate, not this function. Reconcile these before wiring `collate_fn` in.

## Data flow

`ImageRetrievalDataModule.setup()` → instantiate dataset class (downloads W&B artifact, tokenizes captions) → `split_data` random_split → `train_dataloader`/`val_dataloader`. Running anything that touches the datasets requires W&B authentication (`wandb login`) and access to the referenced artifacts.

## Conventions

- New datasets: subclass `ImageRetrievalDataset`, implement `fetch_dataset()`, and register the class in `DATASET_LOOKUP`.
- `datasets/` (local downloads) and `notebooks/` are gitignored/experimental scratch space, not part of the package.
