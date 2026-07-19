import math

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from .embeddings import sinusoidal_embedding, VariableEncoder, FactorEncoder, _mlp
from .layers import BipartiteLayer


class GraphDenoiser(nn.Module):
    """GNN denoiser for diffusion over constraint graphs (TECHNICAL_GUIDE §5).

    Variable nodes are initialized from value + type; factor nodes are
    initialized from type + constraint residual + sinusoidal timestep embedding.
    L rounds of bipartite message passing, then a per-variable output MLP
    predicts the noise eps_hat.
    """

    def __init__(
        self,
        D: int = 256,
        L: int = 4,
        step_dim: int = 64,
        num_var_types: int = 4,
        num_fac_types: int = 4,
    ):
        super().__init__()
        self.step_dim = step_dim

        self.var_encoder = VariableEncoder(D, num_types=num_var_types, step_dim=step_dim)
        self.fac_encoder = FactorEncoder(D, step_dim=step_dim, num_types=num_fac_types)
        self.layers = nn.ModuleList([BipartiteLayer(D) for _ in range(L)])
        self.output_mlp = _mlp(D + step_dim, D)
        self.output_head = nn.Linear(D, 1)
        # Direct skip from a variable's incident-factor constants to its output. The
        # message-passing stack (LayerNorm each round) washes out the magnitude of
        # the constraint constants, so a variable whose value is set by its own
        # constraint (e.g. a linear/near-linear factor) can't be read off from the
        # deep features alone. This linear skip restores that path. New param, so
        # older checkpoints load via strict=False (skip contribution starts small).
        self.const_skip = nn.Linear(1, 1)

        # Precompute sinusoidal frequency bank; avoids recomputation each forward pass.
        half = step_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("_freqs", freqs)

    def _step_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute [B, step_dim] sinusoidal embedding using cached frequencies."""
        args = t.float().unsqueeze(-1) * self._freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        data: HeteroData,
        t: torch.Tensor,
        cas_engine=None,
    ) -> torch.Tensor:
        """
        Args:
            data:       HeteroData with data["variable"].x [n,1], data["factor"].x [m,1],
                        and edge type ("variable","connected_to","factor") carrying
                        edge_index [2,E] and edge_attr [E,1] (coefficient).
            t:          [B] or scalar integer timestep(s).
            cas_engine: optional CASEngine; residuals default to zeros when absent.
        Returns:
            eps_hat [n, 1] — predicted noise per variable node.
        """
        x_var = data["variable"].x
        edge_store = data["variable", "connected_to", "factor"]
        edge_index = edge_store.edge_index
        edge_attr = edge_store.edge_attr

        n_vars = x_var.size(0)
        n_facs = data["factor"].x.size(0)
        device = x_var.device

        t_scalar = t.view(-1)[0].long()
        step_emb_1 = self._step_embedding(t_scalar.unsqueeze(0))  # [1, step_dim]
        step_emb_v = step_emb_1.expand(n_vars, -1)
        step_emb_f = step_emb_1.expand(n_facs, -1)

        # Constant term per factor (encodes the constraint RHS), set by build_heterodata.
        const = data["factor"].x
        if const.size(0) != n_facs:  # robustness for older graphs
            const = torch.zeros(n_facs, 1, device=device)

        # Current residual (constraint violation at x_var). Exact via CAS when given;
        # otherwise the analytic linear residual A·x + const (correct for linear systems).
        if cas_engine is not None:
            res_list = cas_engine.residuals(x_var.detach().squeeze(-1).tolist())
            residuals = torch.tensor(res_list, dtype=torch.float32, device=device).unsqueeze(-1)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = scatter(edge_attr * x_var[src], dst, dim=0, dim_size=n_facs, reduce="sum")
            residuals = agg + const

        fac_type_id = torch.zeros(n_facs, dtype=torch.long, device=device)
        var_type_id = torch.zeros(n_vars, dtype=torch.long, device=device)

        # Gather the constant term of each variable's incident factors so the model
        # can infer that variable's value directly (not only through message passing).
        src, dst = edge_index[0], edge_index[1]
        incident_const = scatter(const[dst], src, dim=0, dim_size=n_vars, reduce="sum")

        h_v = self.var_encoder(x_var, var_type_id, step_emb_v, incident_const)
        h_f = self.fac_encoder(fac_type_id, residuals, step_emb_f, const)

        for layer in self.layers:
            h_v, h_f = layer(h_v, h_f, edge_index, edge_attr)

        out = self.output_head(self.output_mlp(torch.cat([h_v, step_emb_v], dim=-1)))
        return out + self.const_skip(incident_const)
