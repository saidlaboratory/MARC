import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU: in_dim → out_dim → out_dim."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class VariableEncoder(nn.Module):
    """Encode variable nodes: scalar value + optional type id + optional timestep
    embedding → [n, D]. Conditioning on the timestep lets the model know the noise
    level (so it can scale its prediction); without it the variable path is blind to t."""

    def __init__(self, D: int, num_types: int = 4, step_dim: int = 64):
        super().__init__()
        self.mlp = _mlp(1, D)
        self.type_emb = nn.Embedding(num_types, D)
        self.t_proj = nn.Linear(step_dim, D)
        # Condition each variable directly on the constant terms of its incident
        # factors. Without this the constraint constants only reach a variable via
        # message passing, which washes them out and the model collapses to
        # predicting the mean solution. (Standard init: a zero-init path never turns
        # on — the model settles into mean-prediction first. Checkpoints predating
        # this param load via strict=False; the learned-solver polish absorbs the
        # small change in the denoiser output.)
        self.const_proj = nn.Linear(1, D)

    def forward(
        self,
        x: torch.Tensor,
        type_id: torch.Tensor = None,
        step_emb: torch.Tensor = None,
        incident_const: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.mlp(x)
        if type_id is not None:
            h = h + self.type_emb(type_id)
        if step_emb is not None:
            h = h + self.t_proj(step_emb)
        if incident_const is not None:
            h = h + self.const_proj(incident_const)
        return h


class FactorEncoder(nn.Module):
    """Encode factor nodes: type id + current residual + constant term + step
    embedding → [m, D]. The constant term encodes the constraint's RHS; the residual
    is the current violation."""

    def __init__(self, D: int, step_dim: int, num_types: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, D)
        self.mlp = _mlp(D + 2 + step_dim, D)  # +2: residual and constant

    def forward(
        self,
        type_id: torch.Tensor,
        residual: torch.Tensor,
        step_emb: torch.Tensor,
        const: torch.Tensor = None,
    ) -> torch.Tensor:
        te = self.type_emb(type_id)
        if const is None:
            const = torch.zeros_like(residual)
        return self.mlp(torch.cat([te, residual, const, step_emb], dim=-1))
