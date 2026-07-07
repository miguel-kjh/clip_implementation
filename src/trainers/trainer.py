import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_model import CLIPModel
from dataloaders import ImageRetrievalDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

CHECKPOINTS_DIR     = "checkpoints"
IMAGE_ENCODER_ALIAS = "resnet50"
TEXT_ENCODER_ALIAS  = "distilbert-base-uncased"
DATASET_NAME        = "flickr8k"


class ExperimentModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint que, al eliminar un checkpoint, borra también su subcarpeta si queda vacía."""

    def _remove_checkpoint(self, trainer, filepath):
        super()._remove_checkpoint(trainer, filepath)
        folder = os.path.dirname(filepath)
        if (
            folder
            and os.path.normpath(folder) != os.path.normpath(self.dirpath or "")
            and os.path.isdir(folder)
            and not os.listdir(folder)
        ):
            os.rmdir(folder)


def build_logger(logger_type: str, run_name: str):
    if logger_type == "wandb":
        return WandbLogger(project="CLIP", name=run_name, log_model="all")
    if logger_type == "csv":
        return CSVLogger(save_dir="logs", name=run_name)
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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"CLIP-{timestamp}"

    model = CLIPModel(IMAGE_ENCODER_ALIAS, TEXT_ENCODER_ALIAS)
    data_module = ImageRetrievalDataModule(
        dataset_path="datasets/flickr8k",
        dataset_name=DATASET_NAME,
        tokenizer_alias=TEXT_ENCODER_ALIAS,
        lazy_loading=True,
    )

    logger = build_logger(args.logger, experiment_name)

    model_checkpoint = ExperimentModelCheckpoint(
        dirpath=os.path.join(CHECKPOINTS_DIR, experiment_name),
        filename="epoch_{epoch:02d}_val_loss_{val/loss:.4f}/model",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=2,
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
