import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_model import CLIPModel
from dataloaders import ImageRetrievalDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

image_encoder_alias = "resnet50"
text_encoder_alias = "distilbert-base-uncased"
dataset_name = "flickr8k"

model = CLIPModel(image_encoder_alias, text_encoder_alias)
data_module = ImageRetrievalDataModule(
    dataset_path="datasets/flickr8k",
    dataset_name=dataset_name,
    tokenizer_alias=text_encoder_alias,
    lazy_loading=True
)

logger = WandbLogger(project="CLIP", log_model="all")

model_checkpoint = ModelCheckpoint()
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(
    accelerator="gpu", 
    logger=logger,
    max_epochs=20,
    log_every_n_steps=1,
    callbacks=[model_checkpoint, lr_monitor]
)
trainer.fit(model, data_module)