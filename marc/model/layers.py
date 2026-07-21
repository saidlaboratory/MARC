import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from .embeddings import _mlp


class BipartiteLayer(nn.Module):
    """One round of bipartite message passing between variable and factor nodes.

    Update order: factor update (v→f) then variable update (f→v), each with
    a residual connection and LayerNorm. Edge coefficients are injected into
    every message to expose constraint structure.
    """

    def __init__(self, D: int, edge_dim: int = 1):
        super().__init__()
        # ``edge_dim=1`` preserves the original GraphDenoiser/checkpoint
        # contract.  Semantic proposal models can pass a richer polynomial edge
        # signature without maintaining a second copy of the message-passing
        # implementation.

        self.msg_v2f = _mlp(D + D + edge_dim, D)
        self.phi_f   = _mlp(D + D, D)
        self.norm_f  = nn.LayerNorm(D)

        self.msg_f2v = _mlp(D + D + edge_dim, D)
        self.phi_v   = _mlp(D + D, D)
        self.norm_v  = nn.LayerNorm(D)

    def forward(
        self,
        h_v: torch.Tensor,
        h_f: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """
        Args:
            h_v:        [n, D] variable node embeddings
            h_f:        [m, D] factor node embeddings
            edge_index: [2, E] row 0 = variable idx, row 1 = factor idx
            edge_feat:  [E, 1] edge coefficient
        Returns:
            (h_v_new [n, D], h_f_new [m, D])
        """
        src_v = edge_index[0]
        dst_f = edge_index[1]

        # Factor update: aggregate variable messages
        msg_v2f = self.msg_v2f(torch.cat([h_v[src_v], h_f[dst_f], edge_feat], dim=-1))
        agg_f = scatter(msg_v2f, dst_f, dim=0, dim_size=h_f.size(0), reduce="sum")
        h_f_new = self.norm_f(h_f + self.phi_f(torch.cat([h_f, agg_f], dim=-1)))

        # Variable update: aggregate updated factor messages
        msg_f2v = self.msg_f2v(torch.cat([h_f_new[dst_f], h_v[src_v], edge_feat], dim=-1))
        agg_v = scatter(msg_f2v, src_v, dim=0, dim_size=h_v.size(0), reduce="sum")
        h_v_new = self.norm_v(h_v + self.phi_v(torch.cat([h_v, agg_v], dim=-1)))

        return h_v_new, h_f_new
