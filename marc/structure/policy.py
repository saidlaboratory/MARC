"""Trained structure-invention policy (U5): absorbing-D3PM reverse process over
menu slots.

Scope (honest): this is MENU-BASED activation — the policy denoises 2K padded
slots (candidate j owns var slot 2j + factor slot 2j+1) and thereby *picks* which
of K pre-built augmentations to instantiate. Invention here is a slot going
ABSENT -> concrete during the reverse process (TECHNICAL_GUIDE §10); free-form
expression invention is out of scope (§14 ladder — see
``marc.structure.invention_data``).

Composition: the fixed-graph context reuses the P1 encoders/layers
(``VariableEncoder``/``FactorEncoder``/``BipartiteLayer``) with the same
incident-constant/residual conditioning math as ``marc.model.denoiser``; the
output head is the existing ``StructureHead``. Nothing in ``marc/model`` is
modified.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from marc.graph.pyg import build_heterodata
from marc.model.embeddings import FactorEncoder, VariableEncoder, _mlp
from marc.model.layers import BipartiteLayer
from marc.model.structure_head import StructureHead

from .diffusion import keep_schedule
from .invention_data import InventionInstance
from .schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph, SlotType

#: Max number of fixed-graph factors a candidate's u-coefficient vector covers.
M_MAX = 8

#: one-hot corrupted type ++ value ++ role bit ++ t/T ++ pin/4 ++ per-fixed-factor
#: u-coefficient vector zero-padded to M_MAX. No positional features, so the model
#: is permutation-equivariant over the menu.
SLOT_FEATURE_DIM = NUM_SLOT_TYPES + 4 + M_MAX


def slot_features(
    inst: InventionInstance, padded: PaddedGraph, t: int, T: int
) -> torch.Tensor:
    """Per-slot input features [2K, SLOT_FEATURE_DIM] for the policy encoder."""
    K = len(inst.candidates)
    n = 2 * K
    if padded.n_slots != n:
        raise ValueError(f"padded has {padded.n_slots} slots, expected 2K={n}")
    factor_ids = [f.id for f in inst.fixed_graph.factors]
    if len(factor_ids) > M_MAX:
        raise ValueError(f"fixed graph has {len(factor_ids)} factors > M_MAX={M_MAX}")

    base = padded.to_features()  # [n, NUM_SLOT_TYPES + 1]: one-hot ++ value
    role = torch.tensor([j % 2 for j in range(n)], dtype=torch.float32).unsqueeze(-1)
    t_col = torch.full((n, 1), t / max(T - 1, 1), dtype=torch.float32)
    pin = torch.zeros(n, 1, dtype=torch.float32)
    coeff = torch.zeros(n, M_MAX, dtype=torch.float32)
    for j, cand in enumerate(inst.candidates):
        # Expression-defined auxiliaries have no scalar pin in the candidate
        # recipe.  Their solution value is target information and must not leak
        # into a candidate-only feature.
        if cand.defining_expression is None:
            pin[2 * j : 2 * j + 2] = cand.pin_value / 4.0
        for m, fid in enumerate(factor_ids):
            coeff[2 * j : 2 * j + 2, m] = cand.insert_coeffs.get(fid, 0.0)
    return torch.cat([base, role, t_col, pin, coeff], dim=-1)


class StructureEncoder(nn.Module):
    """Fixed-graph context (reused P1 encoders + BipartiteLayers, mean-pooled)
    conditioning per-slot embeddings, with one menu-wide competition round.

    ``ablate_context=True`` zeroes the pooled graph context vector at both of its
    use sites (additive and competition-round) while keeping the parameter set
    identical — the ablation exists to MEASURE whether slot scoring actually
    relies on reading the fixed graph, so state_dicts round-trip either way."""

    def __init__(self, D: int = 64, L: int = 2, step_dim: int = 64,
                 ablate_context: bool = False):
        super().__init__()
        self.step_dim = step_dim
        self.ablate_context = ablate_context
        self.var_encoder = VariableEncoder(D, step_dim=step_dim)
        self.fac_encoder = FactorEncoder(D, step_dim=step_dim)
        self.layers = nn.ModuleList([BipartiteLayer(D) for _ in range(L)])
        self.ctx_proj = nn.Linear(2 * D, D)
        self.slot_mlp = _mlp(SLOT_FEATURE_DIM, D)
        self.comp_mlp = _mlp(3 * D, D)

    def forward(self, data, slot_feats: torch.Tensor) -> torch.Tensor:
        """data: HeteroData of the fixed graph; slot_feats: [2K, SLOT_FEATURE_DIM].
        Returns h_slots [2K, D]."""
        x_var = data["variable"].x
        edge_store = data["variable", "connected_to", "factor"]
        edge_index, edge_attr = edge_store.edge_index, edge_store.edge_attr
        n_vars = x_var.size(0)
        n_facs = data["factor"].x.size(0)
        device = x_var.device
        const = data["factor"].x

        # Mirror marc.model.denoiser: analytic linear residual A·x + const, and
        # incident-constant conditioning per variable, both via scatter.
        src, dst = edge_index[0], edge_index[1]
        residuals = scatter(edge_attr * x_var[src], dst, dim=0, dim_size=n_facs, reduce="sum") + const
        incident_const = scatter(const[dst], src, dim=0, dim_size=n_vars, reduce="sum")

        var_type_id = torch.zeros(n_vars, dtype=torch.long, device=device)
        fac_type_id = torch.zeros(n_facs, dtype=torch.long, device=device)
        # Context is t-independent; the slot features carry t/T.
        step_emb_f = torch.zeros(n_facs, self.step_dim, device=device)

        h_v = self.var_encoder(x_var, var_type_id, None, incident_const)
        h_f = self.fac_encoder(fac_type_id, residuals, step_emb_f, const)
        for layer in self.layers:
            h_v, h_f = layer(h_v, h_f, edge_index, edge_attr)
        ctx = self.ctx_proj(torch.cat([h_v.mean(dim=0), h_f.mean(dim=0)], dim=-1))  # [D]
        if self.ablate_context:
            # Zero (not remove) the context before BOTH uses: parameter set is
            # unchanged, so checkpoints load strictly in either direction.
            ctx = torch.zeros_like(ctx)

        h = self.slot_mlp(slot_feats) + ctx  # [2K, D]
        n = h.size(0)
        # One competition round: every slot sees the menu mean + the graph context.
        h = self.comp_mlp(
            torch.cat([h, h.mean(dim=0, keepdim=True).expand(n, -1),
                       ctx.unsqueeze(0).expand(n, -1)], dim=-1)
        )
        return h


class StructurePolicy(nn.Module):
    """Encoder + the existing StructureHead: predicts p_theta(c_0 | c_t) per slot
    plus a per-slot value (aux value / pin) for committed active slots."""

    def __init__(self, D: int = 64, L: int = 2, K: int = 4, ablate_context: bool = False):
        super().__init__()
        self.K = K
        self.ablate_context = ablate_context
        self.encoder = StructureEncoder(D=D, L=L, ablate_context=ablate_context)
        self.head = StructureHead(D=D, num_slot_types=NUM_SLOT_TYPES)
        # ponytail: heterodata cache keyed by instance id (graphs are immutable;
        # avoids re-sympifying factor expressions every training step)
        self._hd_cache: dict = {}

    def _heterodata(self, inst: InventionInstance):
        data = self._hd_cache.get(inst.id)
        if data is None:
            data = build_heterodata(inst.fixed_graph)
            self._hd_cache[inst.id] = data
        return data

    def forward(
        self, inst: InventionInstance, padded: PaddedGraph, t: int, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (value_pred [2K, 1], slot_logits [2K, NUM_SLOT_TYPES])."""
        device = next(self.parameters()).device
        data = self._heterodata(inst).to(device)
        feats = slot_features(inst, padded, t, T).to(device)
        h = self.encoder(data, feats)
        return self.head(h)


def reverse_sample(
    policy,
    inst: InventionInstance,
    *,
    T: int = 20,
    schedule: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    single_shot: bool = False,
) -> Tuple[PaddedGraph, torch.Tensor]:
    """Absorbing-D3PM ancestral reverse process from an all-ABSENT prior.

    Iterative default: for t = T-1 .. 1 the model gives p_theta(c_0 | c_t); each
    uncommitted slot commits with probability ((ab[t-1]-ab[t])/(1-ab[t])).clamp(0,1)
    — the exact absorbing posterior — to its argmax class (which may be ABSENT, a
    valid commitment). Committed slots freeze; values for committed active slots
    come from value_pred. ab[0]=1 guarantees full commitment by t=1.

    ``single_shot=True``: one forward at t=T-1 on the all-ABSENT input, argmax
    everywhere — the ablation. Returns (final PaddedGraph, last slot_logits [2K, C]).
    """
    if T < 2:
        raise ValueError("T must be >= 2")
    K = len(inst.candidates)
    n = 2 * K
    if schedule is None:
        schedule = keep_schedule(T)
    types = torch.full((n,), ABSENT, dtype=torch.long)
    values = torch.zeros(n, dtype=torch.float32)

    with torch.no_grad():
        if single_shot:
            vp, logits = policy(inst, PaddedGraph(types, values), T - 1, T)
            logits = logits.detach().cpu()
            vals = vp.detach().cpu().squeeze(-1)
            pred = logits.argmax(dim=-1)
            final_vals = torch.where(pred != ABSENT, vals, torch.zeros_like(vals))
            return PaddedGraph(pred, final_vals), logits

        committed = torch.zeros(n, dtype=torch.bool)
        last_logits = None
        for t in range(T - 1, 0, -1):
            vp, logits = policy(inst, PaddedGraph(types.clone(), values.clone()), t, T)
            last_logits = logits.detach().cpu()
            vals = vp.detach().cpu().squeeze(-1)
            denom = 1.0 - float(schedule[t])
            p = 1.0 if denom <= 1e-12 else (float(schedule[t - 1]) - float(schedule[t])) / denom
            p = min(max(p, 0.0), 1.0)
            rand = torch.rand(n, generator=generator)
            newly = (~committed) & (rand < p)
            pred = last_logits.argmax(dim=-1)
            types = torch.where(newly, pred, types)
            values = torch.where(newly & (pred != ABSENT), vals, values)
            committed |= newly
        return PaddedGraph(types, values), last_logits


def chosen_candidate(final: PaddedGraph, logits: torch.Tensor, K: int) -> Optional[int]:
    """Candidate j is chosen iff slot 2j is VARIABLE and slot 2j+1 is FACTOR.

    Ties broken by summed active-class logits; all-ABSENT (no complete pair) -> None.
    """
    types = final.slot_types
    var_t, fac_t = int(SlotType.VARIABLE), int(SlotType.FACTOR)
    hits = [
        j for j in range(K)
        if int(types[2 * j]) == var_t and int(types[2 * j + 1]) == fac_t
    ]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    logits = logits.detach().cpu()
    scores = [float(logits[2 * j, var_t] + logits[2 * j + 1, fac_t]) for j in hits]
    return hits[scores.index(max(scores))]


def predicted_pin(final: PaddedGraph, j: int) -> Optional[float]:
    """Value head's regressed defining value for candidate j: final.values[2j+1]
    iff slot 2j+1 committed FACTOR, else None."""
    # No forward pass needed: reverse_sample already writes value-head outputs
    # into committed slots.
    if int(final.slot_types[2 * j + 1]) != int(SlotType.FACTOR):
        return None
    return float(final.values[2 * j + 1])
