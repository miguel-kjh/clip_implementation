import torch
from torch import nn

class LinearEmbeddingsConnector(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)


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
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.gate_proj  = nn.Linear(embedding_dim, projection_dim)   # W_gate
        self.value_proj = nn.Linear(embedding_dim, projection_dim)   # W_value
        self.act = nn.SiLU()

        self.fc = nn.Linear(projection_dim, projection_dim)          # proyección de salida
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)

        gate  = self.act(self.gate_proj(x))   # SiLU(W_gate · x)
        value = self.value_proj(x)            # W_value · x

        x = gate * value                      # producto de Hadamard
        x = self.fc(x)
        x = self.dropout(x)
        return x