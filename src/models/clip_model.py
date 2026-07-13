from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoder, TextEncoder
from .embeddings_connectors.embeddings_conector import build_embeddings_connector


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        image_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        text_encoder_pooling: str = "auto",
        image_embedding_dims: Optional[int] = None,
        text_embedding_dims: Optional[int] = None,
        projection_dims: int = 256,
        # Con dropout=0.0 el entrenamiento se queda en la solución degenerada de
        # _compute_losses (todos los embeddings iguales ⇒ loss = ln(batch_size)).
        dropout: float = 0.1,
        temperature: float = 1.0,
        image_connector: str = "mlp",
        text_connector: str = "mlp",
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias,
            trainable=text_encoder_trainable,
            pooling=text_encoder_pooling,
        )

        if image_embedding_dims is None:
            image_embedding_dims = self.image_encoder.embedding_dim
        if text_embedding_dims is None:
            text_embedding_dims = self.text_encoder.embedding_dim

        self.image_projection, image_out_dims = build_embeddings_connector(
            image_connector,
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection, text_out_dims = build_embeddings_connector(
            text_connector,
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        # The contrastive loss dots image against text embeddings, so both branches
        # must land in the same space. Without a connector the encoder dim passes
        # straight through, which rarely matches the other branch.
        if image_out_dims != text_out_dims:
            raise ValueError(
                f"Image and text embeddings must share a dimension, got "
                f"{image_out_dims} (image encoder '{image_encoder_alias}' "
                f"[{image_embedding_dims}] + connector '{image_connector}') vs "
                f"{text_out_dims} (text encoder '{text_encoder_alias}' "
                f"[{text_embedding_dims}] + connector '{text_connector}'). "
                f"With connector 'none' the encoder dimension passes through "
                f"unchanged, so both encoders must already emit the same dimension."
            )

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature

    def _compute_losses(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

    def encode_image(self, images):
        """Embeddings de imagen en el espacio compartido. Sin normalizar L2."""
        return self.image_projection(self.image_encoder(images))

    def encode_text(self, input_ids, attention_mask):
        """Embeddings de texto en el espacio compartido. Sin normalizar L2."""
        return self.text_projection(
            self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        )

    def forward(self, batch):
        image_embeddings = self.encode_image(batch["image"])
        text_embeddings = self.encode_text(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = self._compute_losses(image_embeddings, text_embeddings)
        return loss.mean()
