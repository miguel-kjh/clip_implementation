import torch
from torch import nn

class LinearEmbeddingsConnector(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.dropout(x)
        return self.layer_norm(x)


class MLPEmbeddingsConnector(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)
    
class SwiGLUEmbeddingsConnector(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.gate_proj  = nn.Linear(embedding_dim, projection_dim)   # W_gate
        self.value_proj = nn.Linear(embedding_dim, projection_dim)   # W_value
        self.act = nn.SiLU()

        self.fc = nn.Linear(projection_dim, projection_dim)          # proyección de salida
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        gate  = self.act(self.gate_proj(x))   # SiLU(W_gate · x)
        value = self.value_proj(x)            # W_value · x

        x = gate * value                      # producto de Hadamard
        x = self.fc(x)
        x = self.dropout(x)
        return self.layer_norm(x)


CONNECTOR_LOOKUP = {
    "linear": LinearEmbeddingsConnector,
    "mlp": MLPEmbeddingsConnector,
    "swiglu": SwiGLUEmbeddingsConnector,
    "none": None,
}


def build_embeddings_connector(
    name: str, embedding_dim: int, projection_dim: int, dropout: float
):
    """Construye un conector y devuelve (módulo, dimensión de salida)."""
    key = name.lower()
    if key not in CONNECTOR_LOOKUP:
        raise ValueError(
            f"Unknown embeddings connector '{name}'. "
            f"Valid options: {', '.join(CONNECTOR_LOOKUP)}"
        )

    if key == "none":
        return nn.Identity(), embedding_dim

    if key == "linear":
        return LinearEmbeddingsConnector(embedding_dim, projection_dim, dropout), projection_dim

    connector = CONNECTOR_LOOKUP[key](
        embedding_dim=embedding_dim, projection_dim=projection_dim, dropout=dropout
    )
    return connector, projection_dim 