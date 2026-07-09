import os
import sys
import argparse
import itertools
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_model import CLIPModel
from models.checkpoint import save_hparams
from models.embeddings_connectors import CONNECTOR_LOOKUP
from dataloaders.flickr import Flickr8kDataset, Flickr30kDataset


IMAGE_ENCODER_ALIAS = "resnet50"
TEXT_ENCODER_ALIAS  = "distilbert-base-uncased"

DATASET_LOOKUP = {
    "flickr8k": Flickr8kDataset,
    "flickr30k": Flickr30kDataset,
}

def seed_everything(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def build_experiment_dir(args):
    # Aliases like "distilbert-base-uncased" may be namespaced ("org/model"),
    # which would otherwise be read as a nested path.
    image_alias = IMAGE_ENCODER_ALIAS.replace("/", "-")
    text_alias = TEXT_ENCODER_ALIAS.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiment_name = (
        f"{image_alias}_{text_alias}"
        f"_img{args.image_connector}_txt{args.text_connector}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_{args.dataset_name}"
        f"_{timestamp}"
    )
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def build_model_hparams(args):
    """Argumentos con los que se construye el CLIPModel. Se guardan tal cual para
    poder reconstruirlo desde el checkpoint (ver models/checkpoint.py)."""
    return {
        "image_encoder_alias": IMAGE_ENCODER_ALIAS,
        "text_encoder_alias": TEXT_ENCODER_ALIAS,
        "image_connector": args.image_connector,
        "text_connector": args.text_connector,
        "projection_dims": args.projection_dims,
        "dropout": args.dropout,
        "temperature": args.temperature,
    }


def build_hparams(args, model_hparams):
    return {
        "model": model_hparams,
        "data": {
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
            "size": args.size,
            "max_length": args.max_length,
            "val_split": args.val_split,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "image_encoder_lr": args.image_encoder_lr,
            "text_encoder_lr": args.text_encoder_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "checkpoint": args.output,
        },
    }


def build_loaders(args, tokenizer):
    dataset = DATASET_LOOKUP[args.dataset_name](
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        target_size=args.size,
        max_length=args.max_length,
    )
    val_length = int(args.val_split * len(dataset))
    train_length = len(dataset) - val_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    parser = argparse.ArgumentParser(description="Train the CLIP model (pure PyTorch).")
    parser.add_argument("--dataset-name", choices=list(DATASET_LOOKUP), default="flickr8k")
    parser.add_argument("--dataset-path", default="datasets/flickr8k")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--image-connector", choices=list(CONNECTOR_LOOKUP), default="mlp")
    parser.add_argument("--text-connector", choices=list(CONNECTOR_LOOKUP), default="mlp")
    parser.add_argument("--projection-dims", type=int, default=256)
    # Ojo: con dropout=0 los embeddings colapsan (loss se clava en ln(batch_size)).
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--image-encoder-lr", type=float, default=1e-4)
    parser.add_argument("--text-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.8)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--output", default="best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = build_experiment_dir(args)
    checkpoint_path = os.path.join(experiment_dir, args.output)
    print(f"Experiment directory: {experiment_dir}")

    model_hparams = build_model_hparams(args)
    # Al empezar, no al terminar: así un run interrumpido sigue siendo legible.
    save_hparams(experiment_dir, build_hparams(args, model_hparams))

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ALIAS)
    train_loader, valid_loader = build_loaders(args, tokenizer)

    model = CLIPModel(**model_hparams).to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": args.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": args.text_encoder_lr},
        {
            "params": itertools.chain(
                model.image_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor
    )
    step = "epoch"

    best_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_epoch(model, train_loader, optimizer, lr_scheduler, step, device)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, device)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved Best Model to {checkpoint_path}!")

        lr_scheduler.step(valid_loss.avg)


if __name__ == "__main__":
    main()
