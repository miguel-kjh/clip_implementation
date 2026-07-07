import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_model import CLIPModel
from dataloaders import ImageRetrievalDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

CHECKPOINTS_DIR     = "checkpoints"
IMAGE_ENCODER_ALIAS = "resnet50"
TEXT_ENCODER_ALIAS  = "distilbert-base-uncased"
DATASET_NAME        = "flickr8k"


def build_logger(logger_type: str):
    if logger_type == "wandb":
        return WandbLogger(project="CLIP", log_model="all")
    if logger_type == "csv":
        return CSVLogger(save_dir="logs", name="CLIP")
    raise ValueError(f"Unknown logger type: {logger_type!r} (use 'wandb' or 'csv')")


def main():
    parser = argparse.ArgumentParser(description="Train the CLIP model.")
    parser.add_argument(
        "--logger",
        choices=["wandb", "csv"],
        default="csv",
        help="Logger backend to use: 'wandb' or a normal 'csv' logger.",
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--accelerator", default="gpu")
    args = parser.parse_args()

    model = CLIPModel(IMAGE_ENCODER_ALIAS, TEXT_ENCODER_ALIAS)
    data_module = ImageRetrievalDataModule(
        dataset_path="datasets/flickr8k",
        dataset_name=DATASET_NAME,
        tokenizer_alias=TEXT_ENCODER_ALIAS,
        lazy_loading=True,
    )

    logger = build_logger(args.logger)

    model_checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename="clip-{epoch:02d}-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator=args.accelerator,
        logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[model_checkpoint, lr_monitor],
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
