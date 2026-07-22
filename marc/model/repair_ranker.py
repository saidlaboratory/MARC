"""Candidate-conditioned ranking of structural repairs.

The value-denoising model tries to encode one fixed graph and numerically solve it.
The structure policy instead receives a menu of graph augmentations.  This module
scores the graph *after* each candidate repair is applied, then competes the scores
with a listwise softmax.  It therefore learns the discrete decision MARC needs while
leaving all continuous solving to the classical backend.
"""

from __future__ import annotations

import math
from typing import Sequence

import sympy as sp
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from marc.graph.semantics import EDGE_FEATURE_DIM, FACTOR_FEATURE_DIM
from marc.structure.invention_data import Candidate, InventionInstance

from .embeddings import _mlp
from .layers import BipartiteLayer


M_MAX = 8
CANDIDATE_FEATURE_DIM = 2 * M_MAX + 7


class GraphRepairRanker(nn.Module):
    """Assign one compatibility score to every candidate-augmented graph."""

    def __init__(self, D: int = 96, L: int = 3):
        super().__init__()
        self.var_encoder = _mlp(1, D)
        self.fac_encoder = _mlp(FACTOR_FEATURE_DIM, D)
        self.layers = nn.ModuleList(
            [BipartiteLayer(D, edge_dim=EDGE_FEATURE_DIM) for _ in range(L)]
        )
        # mean + max over both node types, plus log node/factor counts
        self.score = nn.Sequential(
            nn.Linear(4 * D + 2, D),
            nn.SiLU(),
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, 1),
        )

    def forward(self, data) -> torch.Tensor:
        h_v = self.var_encoder(data["variable"].x)
        h_f = self.fac_encoder(data["factor"].x)
        store = data["variable", "connected_to", "factor"]
        for layer in self.layers:
            h_v, h_f = layer(h_v, h_f, store.edge_index, store.edge_attr)

        vb = getattr(data["variable"], "batch", None)
        fb = getattr(data["factor"], "batch", None)
        if vb is None:
            vb = torch.zeros(h_v.size(0), dtype=torch.long, device=h_v.device)
        if fb is None:
            fb = torch.zeros(h_f.size(0), dtype=torch.long, device=h_f.device)
        B = int(max(vb.max().item(), fb.max().item())) + 1
        pools = [
            scatter(h_v, vb, dim=0, dim_size=B, reduce="mean"),
            scatter(h_v, vb, dim=0, dim_size=B, reduce="max"),
            scatter(h_f, fb, dim=0, dim_size=B, reduce="mean"),
            scatter(h_f, fb, dim=0, dim_size=B, reduce="max"),
        ]
        counts = torch.stack([
            torch.bincount(vb, minlength=B),
            torch.bincount(fb, minlength=B),
        ], dim=-1).to(h_v.dtype).log1p()
        return self.score(torch.cat([*pools, counts], dim=-1)).squeeze(-1)


class CandidateOnlyRanker(nn.Module):
    """No-problem-context control: score only the augmentation recipe."""

    def __init__(self, D: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, D),
            nn.SiLU(),
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def batch_candidate_graphs(instances: Sequence[InventionInstance], build_fn) -> Batch:
    """Flatten menus in instance-major order into one PyG batch."""
    return Batch.from_data_list([
        build_fn(candidate.apply(inst.fixed_graph))
        for inst in instances
        for candidate in inst.candidates
    ])


def candidate_features(inst: InventionInstance, candidate: Candidate) -> torch.Tensor:
    """Fixed-width recipe features for the no-context control.

    These expose the candidate's pin, insertion support/coefficients, and defining
    expression shape, but deliberately no constants or topology from the problem
    being repaired.
    """
    factor_ids = [f.id for f in inst.fixed_graph.factors]
    if len(factor_ids) > M_MAX:
        raise ValueError(f"fixed graph has {len(factor_ids)} factors > M_MAX={M_MAX}")
    support = [0.0] * M_MAX
    coeffs = [0.0] * M_MAX
    for i, fid in enumerate(factor_ids):
        if fid in candidate.insert_coeffs:
            support[i] = 1.0
            coeffs[i] = float(candidate.insert_coeffs[fid]) / 2.0

    present = float(candidate.defining_expression is not None)
    degree = terms = has_cross = has_square = constant = 0.0
    if candidate.defining_expression is not None:
        expr = sp.sympify(candidate.defining_expression)
        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        poly = sp.Poly(expr, *symbols)
        degree = float(poly.total_degree()) / 4.0
        terms = math.log1p(len(poly.terms()))
        has_cross = float(any(sum(e > 0 for e in m) >= 2 for m, _ in poly.terms()))
        has_square = float(any(any(e >= 2 for e in m) for m, _ in poly.terms()))
        constant = math.copysign(
            math.log1p(abs(float(poly.eval({s: 0 for s in symbols})))),
            float(poly.eval({s: 0 for s in symbols})),
        )
    values = [
        (float(candidate.pin_value) / 4.0
         if candidate.defining_expression is None else 0.0),
        sum(support) / M_MAX,
        present,
        degree,
        terms,
        has_cross + has_square,
        constant,
        *support,
        *coeffs,
    ]
    return torch.tensor(values, dtype=torch.float32)
