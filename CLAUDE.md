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

Key flags (see `main()` in `src/trainers/trainer.py` for the full list): `--dataset-name` (`flickr8k`/`flickr30k`), `--dataset-path` (local dir), `--epochs`, `--batch-size`, `--size` (image resize), `--max-length` (caption tokens), `--image-connector`/`--text-connector`, and three separate learning rates: `--image-encoder-lr`, `--text-encoder-lr`, `--head-lr`. The best checkpoint (lowest val loss) is saved to `--output-dir/<experimento>/--output` (default `checkpoints/<experimento>/best.pt`).

The image/text encoder **aliases are hardcoded** at the top of `trainer.py` (`IMAGE_ENCODER_ALIAS = "resnet50"`, `TEXT_ENCODER_ALIAS = "distilbert-base-uncased"`), not exposed as CLI flags. Change them there.

### Checkpoints (`checkpoints/<experimento>/`)

Each run writes an experiment dir containing:

- `hparams.json` — written **at the start** of the run (so an interrupted run stays readable) by `save_hparams` in `src/models/checkpoint.py`. Three sections; only `model` is needed to rebuild the net, and it is exactly the kwargs `trainer.py` passes to `CLIPModel(**model_hparams)` (`build_model_hparams`), so the two cannot drift.
- `best.pt` — the lowest-val-loss `state_dict` (`torch.save(model.state_dict())`, filename from `--output`).

Load with `load_clip_model(experiment_dir)` (`src/models/checkpoint.py`), which rebuilds `CLIPModel` from `hparams["model"]` (forcing `image_encoder_pretrained=False`, since the `state_dict` overwrites the ImageNet weights anyway) and returns `(model.eval(), hparams)`. `list_experiments(checkpoints_dir)` enumerates runs and flags the un-loadable ones. **A dir without `hparams.json` cannot be loaded** — the connector choice isn't recoverable from the `state_dict`; retrain it. Checkpoints from before this convention are in this state.

There is no test suite, linter config, or build step in the repo.

## Evaluation (`src/eval/`)

Val loss cannot tell a working model from a collapsed one (see the `--dropout 0` caveat below), so retrieval quality and embedding geometry are measured separately:

```bash
python src/eval/evaluate.py --experiment-dir checkpoints/<exp> --gallery-size 1000
```

Everything else (dataset, image size, `max_length`, `val_split`, seed) is read from the experiment's `hparams.json`. Results are printed and written to `metrics.json` in the experiment dir. `--untrained` re-runs with random weights as a chance control (and writes nothing).

**Retrieval:** Recall@1/5/10, median/mean rank and MRR, both text→image and image→text. The gallery is the *unique images* of the subset; image→text counts a hit if **any** of an image's captions lands in the top-K (standard Flickr protocol). Embeddings are L2-normalized before ranking, unlike in the loss. **Ties are broken pessimistically** (`rank = #{sim >= sim_positive} - 1`): with strict `>`, a collapsed model whose similarities are all identical scores Recall@1 = 100%, because no distractor scores *strictly* higher.

**Collapse** (`src/eval/collapse.py`), per modality: mean off-diagonal cosine, effective rank (RankMe), top-1/top-10 PCA variance, mean per-dim std, embedding norm, and uniformity; plus cross-modal alignment and modality gap.

- **The effective rank is the reliable collapse signal, not the mean cosine.** A healthy model here has mean cosine ≈ 0.81–0.91 — the CLIP *cone effect*, where all embeddings share a large mean component — while a collapsed one has ≈ 1.0. Both look "high". The effective rank separates them cleanly: ≈ 40–75 (of 256) healthy vs ≈ 1.0 collapsed. `evaluate.py` reports a cone-effect note rather than a collapse warning above 0.9 cosine when the rank is healthy.
- The effective rank also catches *partial dimensional* collapse (embeddings spanning 8 of 256 dims) that the mean cosine misses entirely — in that regime the mean cosine reads ≈ 0.

### Two splits, and why the numbers are not comparable by default

`build_loaders` splits at the **caption** level, so ~4 of every image's 5 captions land in train and the image itself is seen during training. `--split val` reproduces exactly that subset (it replays the trainer's seed → build → `random_split` sequence, which depends on nothing else consuming torch's global RNG in between); `--split val-disjoint` groups by image first, so no val image appears in train. Both run by default.

Comparing their Recall requires care, and the script warns about both traps:

- **Gallery size dominates Recall@K.** Ungated, `val` has a 5421-image gallery and `val-disjoint` 1618, which alone swings text→image R@1 from 4.96% to 14.70%. Always pass `--gallery-size` (1000 is the Flickr convention) to compare.
- **Captions per image differ** (1.5 for `val` vs 5.0 for `val-disjoint`), so **image→text** is structurally easier on the disjoint split (more correct candidates). Only text→image is comparable across splits.

At an equal 1000-image gallery, the caption-level leakage turns out **not** to inflate text→image Recall much (1-epoch checkpoint: 16.0% on `val` vs 18.6% on `val-disjoint`).

Reference numbers, `--gallery-size 1000`, `val-disjoint`, text→image R@1: 1 epoch ≈ 18.6%, 4 epochs ≈ 40.3%. Chance is 0.10%; a chance-level result means collapse, not "undertrained".

## Architecture

### Data loading (local disk — NOT W&B)

Datasets are read from **local directories** passed as `dataset_path`. (An earlier design fetched data from Weights & Biases artifacts; that is gone — `wandb` remains in requirements but the training path does not use it.)

- `src/dataloaders/ImageRetrievalDataset.py` — `ImageRetrievalDataset`, abstract base `Dataset`. `__init__` calls the subclass hook `fetch_dataset()`, tokenizes **all** captions up front with a HuggingFace tokenizer (`padding=True, truncation=True`, `return_tensors='pt'`), and builds Albumentations resize+normalize transforms. `__getitem__` returns a dict: tokenizer outputs (`input_ids`, `attention_mask`, ...) sliced per-index, plus `image` (CHW float tensor loaded via `cv2`, BGR→RGB) and `caption` (raw string). Subclasses implement `fetch_dataset()` → `(image_files, captions)`.
- `src/dataloaders/flickr.py` — `Flickr8kDataset` / `Flickr30kDataset`. Each reads local annotation files and `assert`s every image path exists. Flickr8k: `captions.txt` (comma-separated, columns `image`/`caption`, images under `Images/`). Flickr30k: `results.csv` (`|`-separated, columns `image_name`/` comment` — note the leading space; `dropna()` applied; images under `flickr30k_images/`).

### Model (`src/models/`)

- `src/models/clip_model.py` — `CLIPModel` (a plain `nn.Module`). `forward(batch)` runs both encoders, projects each into `projection_dims`, and **returns the scalar loss directly** (not embeddings). To get embeddings use `encode_image(images)` / `encode_text(input_ids, attention_mask)`; neither L2-normalizes, so normalize before computing cosines. Encoder aliases and dims are constructor args (`image_embedding_dims=2048` for resnet50, `text_embedding_dims=768` for distilbert, `projection_dims=256`).
- `src/models/encoders/image_encoders.py` — `ImageEncoder` wraps `timm.create_model(..., num_classes=0, global_pool="avg")` to get a pooled feature vector.
- `src/models/encoders/text_encoders.py` — `TextEncoder` wraps `transformers.AutoModel` and takes the **CLS token** (`last_hidden_state[:, 0, :]`) as the sentence embedding.
- `src/models/embeddings_connectors/embeddings_conector.py` (filename misspelled) — three projection heads, selected per-branch via `--image-connector` / `--text-connector` (`build_embeddings_connector`): `LinearEmbeddingsConnector`, `MLPEmbeddingsConnector` (Linear→GELU→Linear→dropout + residual + LayerNorm) and `SwiGLUEmbeddingsConnector`. Only the MLP head ends in a LayerNorm; with `swiglu` the embedding norm is unbounded and the (unnormalized) loss diverges — a 1-epoch flickr8k run went from loss 3.8 to 40.

### Loss

`CLIPModel._compute_losses` is the distillation-style CLIP loss (à la the Moein Shariatnia / Keras CLIP tutorial, mirrored in `notebooks/OpenAI_CLIP_simple_implementation.ipynb`), **not** the symmetric InfoNCE from the original paper: soft targets are `softmax((image_sim + text_sim)/2 * temperature)`, and image/text cross-entropy losses are averaged. Note: embeddings are **not L2-normalized** before the dot products, and `temperature` is *multiplied* into the targets but *divides* the logits — deliberate quirks of this formulation, so preserve them unless intentionally changing the loss.

**Do not train with `--dropout 0`.** Because the targets are built from the embeddings themselves, "every embedding in the batch is identical" trivially satisfies the loss (uniform logits vs uniform targets) at exactly `ln(batch_size)` — 3.466 for `--batch-size 32`. Without dropout the run falls into that plateau and stays: a 4-epoch flickr8k run gave text-text cosine 0.9999 and text→image Recall@1 of 0.10%, i.e. chance. The reference notebook's `CFG.dropout = 0.1` breaks the symmetry; with it, 1 epoch reaches val loss 0.876 and Recall@1 ≈ 18%. Hence `--dropout` defaults to 0.1 in `trainer.py` and in `CLIPModel`. A collapsed checkpoint is easy to spot: the cosine between two unrelated queries' text embeddings is ~1.0. Run `src/eval/evaluate.py` to check — it flags collapse off the effective rank, which is ≈ 1.0 in that state.

## Known dead / orphaned code

These exist but are **not** on the training path — reconcile before wiring them in:

- `src/dataloaders/ImageRetrievalDataModule.py` — a `LightningDataModule`. Fully functional but **unused**: `trainer.py` builds its own `DataLoader`s (`build_loaders`) and has its own copy of `DATASET_LOOKUP`. Only reach for the DataModule if reintroducing a Lightning training path.
- `src/dataloaders/utils.py` — `collate_fn` expects **tuple-shaped** items `(img, caption, input_id, attention_mask)`, but `__getitem__` returns a **dict**. It is imported by the DataModule but never passed to a `DataLoader`; the default collate is used everywhere. It also has a leftover `print(time.time() - start)`.
- `logs/` — CSV metrics + `hparams.yaml` from earlier PyTorch-Lightning `CSVLogger` runs. The current `trainer.py` does not write here.

## Conventions

- New datasets: subclass `ImageRetrievalDataset`, implement `fetch_dataset()`, and register in **both** `DATASET_LOOKUP` dicts (`trainer.py` and `ImageRetrievalDataModule.py`) if you want it usable from either.
- New projection heads go in `embeddings_conector.py`; to use one, swap the class in `CLIPModel.__init__`.
- `datasets/` (local data) and `notebooks/` are gitignored experimental scratch space, not part of the package.
