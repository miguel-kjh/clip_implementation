import torch
import transformers
from torch import nn

# Tipos de modelo (config.model_type) que son decoder-only: no tienen token CLS,
# el token informativo es el último no-padding.
DECODER_MODEL_TYPES = {
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "llama",
    "mistral",
    "opt",
    "bloom",
    "falcon",
}

VALID_POOLINGS = {"auto", "cls", "mean", "last"}


class TextEncoder(nn.Module):
    """Encapsula un modelo de texto de HuggingFace y produce un embedding de frase
    `[B, hidden]`. La estrategia de pooling depende de la familia del modelo:

    - encoder tipo BERT/DistilBERT -> token CLS (`cls`);
    - decoder-only tipo GPT        -> último token no-padding (`last`);
    - sentence-transformers        -> media enmascarada de los tokens (`mean`).

    Con `pooling="auto"` se deriva de la config del modelo; se puede forzar con
    cualquiera de `{"cls", "mean", "last"}`.
    """

    def __init__(
        self, model_name: str, trainable: bool = True, pooling: str = "auto"
    ) -> None:
        super().__init__()

        if pooling not in VALID_POOLINGS:
            raise ValueError(
                f"pooling debe ser uno de {sorted(VALID_POOLINGS)}, no {pooling!r}."
            )

        self.model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.pooling = self._resolve_pooling(pooling, model_name)
        self.embedding_dim = self.model.config.hidden_size
        print(
            f"TextEncoder '{model_name}': pooling '{self.pooling}'"
            + ("" if pooling != "auto" else " (auto)")
        )

    def _resolve_pooling(self, pooling: str, model_name: str) -> str:
        if pooling != "auto":
            return pooling

        config = self.model.config
        model_type = getattr(config, "model_type", "")
        is_decoder = getattr(config, "is_decoder", False)

        if is_decoder or model_type in DECODER_MODEL_TYPES:
            return "last"
        if "sentence-transformers" in model_name.lower():
            return "mean"
        return "cls"

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts

        if self.pooling == "last":
            # Último token no-padding por fila; robusto con padding a izquierda o derecha.
            last_idx = attention_mask.sum(dim=1).long() - 1
            last_idx = last_idx.clamp(min=0)
            batch_idx = torch.arange(
                last_hidden_state.size(0), device=last_hidden_state.device
            )
            return last_hidden_state[batch_idx, last_idx, :]

        raise ValueError(f"pooling no soportado: {self.pooling!r}")

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(output.last_hidden_state, attention_mask)
