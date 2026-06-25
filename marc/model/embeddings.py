import math

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU: in_dim → out_dim → out_dim."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps [B] to sinusoidal embeddings [B, dim].

    Uses the transformer convention: freq_i = 1 / 10000^(2i / dim),
    with half the dimensions as sine and half as cosine.
    """
    assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VariableEncoder(nn.Module):
    """Encode variable nodes: scalar value + optional type id → [n, D]."""

    def __init__(self, D: int, num_types: int = 4):
        super().__init__()
        self.mlp = _mlp(1, D)
        self.type_emb = nn.Embedding(num_types, D)

    def forward(self, x: torch.Tensor, type_id: torch.Tensor = None) -> torch.Tensor:
        h = self.mlp(x)
        if type_id is not None:
            h = h + self.type_emb(type_id)
        return h


class FactorEncoder(nn.Module):
    """Encode factor nodes: type id + residual + step embedding → [m, D]."""

    def __init__(self, D: int, step_dim: int, num_types: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, D)
        self.mlp = _mlp(D + 1 + step_dim, D)

    def forward(
        self,
        type_id: torch.Tensor,
        residual: torch.Tensor,
        step_emb: torch.Tensor,
    ) -> torch.Tensor:
        te = self.type_emb(type_id)
        return self.mlp(torch.cat([te, residual, step_emb], dim=-1))
