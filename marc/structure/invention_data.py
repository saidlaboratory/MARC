"""Menu-based structure-invention data (U5).

Scope ladder (TECHNICAL_GUIDE §14 — "simplest thing that exhibits the phenomenon";
§10 defines invention as an ABSENT slot going concrete during denoising):

  1. THIS MODULE: menu-based activation. Given an unsolvable fixed graph and a
     procedurally generated menu of K candidate augmentations (aux variable ``u``
     with a defining factor + insertion coefficients), exactly one of which makes
     the graph solvable, the policy must pick the right invention.
  2. OUT OF SCOPE: free-form expression invention (emitting an arbitrary defining
     expression). The menu formulation is the honest preliminary rung.

Each :class:`Candidate` is a complete augmentation recipe; :func:`build_menu`
certifies every distractor inconsistent via an exact symbolic rank check, so
"exactly one solvable option" is a theorem about the data, not an assumption.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import torch

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

from .schema import ABSENT, PaddedGraph, SlotType


@dataclass(frozen=True)
class Candidate:
    """One augmentation recipe: add latent ``aux_var`` pinned to ``pin_value`` and
    insert it (with the given coefficients) into a subset of fixed-graph factors."""

    aux_var: str                      # always "u"
    pin_value: float                  # v in the defining factor "u - (v)"
    insert_coeffs: Dict[str, float]   # fixed-graph factor_id -> coefficient of u

    def apply(self, fixed: FactorGraph) -> FactorGraph:
        """Build the augmented graph. Pure — ``fixed`` is not mutated."""
        variables = [VariableNode(v.id, v.value) for v in fixed.variables]
        factors: List[FactorNode] = []
        edges = [Edge(e.variable_id, e.factor_id, e.coefficient) for e in fixed.edges]
        known = {f.id for f in fixed.factors}
        unknown = set(self.insert_coeffs) - known
        if unknown:
            raise ValueError(f"insert_coeffs reference unknown factors: {sorted(unknown)}")
        if "aux" in known or any(v.id == self.aux_var for v in fixed.variables):
            raise ValueError("fixed graph already contains an 'aux' factor or the aux variable")
        for f in fixed.factors:
            c = self.insert_coeffs.get(f.id)
            if c is None:
                factors.append(FactorNode(f.id, f.expression))
            else:
                factors.append(FactorNode(f.id, f"({f.expression}) + ({c})*{self.aux_var}"))
                edges.append(Edge(self.aux_var, f.id, float(c)))
        variables.append(VariableNode(self.aux_var))
        factors.append(FactorNode("aux", f"{self.aux_var} - ({self.pin_value})"))
        edges.append(Edge(self.aux_var, "aux", 1.0))
        return FactorGraph(variables=variables, factors=factors, edges=edges)


@dataclass
class InventionInstance:
    """A fixed (inconsistent) graph plus a K-candidate menu with one gold fix."""

    id: str
    family: str
    seed: int
    fixed_graph: FactorGraph
    candidates: List[Candidate]        # length K, gold at gold_idx (shuffled per seed)
    gold_idx: int
    aux_value: float                   # value of u in the augmented gold solution
    solution: Dict[str, float] = field(default_factory=dict)  # augmented gold incl. u


def _is_inconsistent(graph: FactorGraph) -> bool:
    """Exact certificate that a *linear* factor graph has no solution.

    rank(A) < rank([A|b]) for the system A x = b read off the factor expressions.
    """
    # ponytail: self-contained ~10-line certificate; deliberately NOT imported from
    # marc/data/aux_required.py so this module works on main today.
    syms = [sp.Symbol(v.id) for v in graph.variables]
    exprs = [sp.sympify(f.expression) for f in graph.factors]
    A, b = sp.linear_eq_to_matrix(exprs, syms)
    return A.rank() < A.row_join(b).rank()


def _candidate_key(c: Candidate) -> Tuple:
    return (float(c.pin_value), tuple(sorted(c.insert_coeffs.items())))


def build_menu(
    fixed: FactorGraph,
    gold_candidate: Candidate,
    K: int,
    rng: random.Random,
    hard_negatives: bool = True,
) -> Tuple[List[Candidate], int]:
    """Assemble a K-candidate menu around the gold fix; returns (menu, gold_idx).

    Distractors: random nonempty factor subsets + coefficients from {-2,-1,1,2} +
    integer pin in [-4,4], each CERTIFIED inconsistent (accidental-consistent or
    gold-duplicate draws are resampled). When ``hard_negatives``, one distractor
    shares the gold insert_coeffs but has a wrong pin value — it forces the policy
    to read constants, not just insertion topology.
    """
    if K < 2:
        raise ValueError("K must be >= 2 (gold + at least one distractor)")
    factor_ids = [f.id for f in fixed.factors]
    menu = [gold_candidate]
    seen = {_candidate_key(gold_candidate)}

    if hard_negatives:
        for _ in range(200):
            pin = float(rng.randint(-4, 4))
            cand = Candidate(gold_candidate.aux_var, pin, dict(gold_candidate.insert_coeffs))
            if _candidate_key(cand) in seen:
                continue
            if _is_inconsistent(cand.apply(fixed)):
                menu.append(cand)
                seen.add(_candidate_key(cand))
                break
        else:
            raise RuntimeError("could not certify a hard negative in 200 draws")

    attempts = 0
    while len(menu) < K:
        attempts += 1
        if attempts > 500:
            raise RuntimeError("could not fill the menu with certified distractors")
        subset = [fid for fid in factor_ids if rng.random() < 0.5]
        if not subset:
            continue
        cand = Candidate(
            gold_candidate.aux_var,
            float(rng.randint(-4, 4)),
            {fid: float(rng.choice((-2, -1, 1, 2))) for fid in subset},
        )
        if _candidate_key(cand) in seen:
            continue
        if not _is_inconsistent(cand.apply(fixed)):
            continue  # accidental-consistent draw -> resample
        menu.append(cand)
        seen.add(_candidate_key(cand))

    rng.shuffle(menu)
    return menu, menu.index(gold_candidate)


# --- toy-variant resampler (main-only data source) ----------------------------

#: family -> (variable names,
#:            base factors as (factor_id, {var: coeff}),
#:            factor ids the latent u touches — mirrors the 3 structure_toys patterns:
#:            isolated offset / single-variable coupling / shared across measurements).
_TEMPLATES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, Dict[str, int]]], Tuple[str, ...]]] = {
    "toy1": (("x", "y"),
             [("eq1", {"x": 1, "y": 1}), ("eq2", {"x": 1, "y": -1}), ("eq3", {"x": 1})],
             ("eq1", "eq2", "eq3")),
    "toy2": (("x", "y", "z"),
             [("eq1", {"x": 1, "y": 1, "z": 1}), ("eq2", {"x": 1, "y": -1}),
              ("eq3", {"y": 1, "z": -1}), ("eq4", {"x": 1})],
             ("eq4",)),
    "toy3": (("x", "y"),
             [("eq1", {"x": 1}), ("eq2", {"y": 1}), ("eq3", {"x": 1, "y": 1})],
             ("eq1", "eq2")),
}

FAMILIES: Tuple[str, ...] = tuple(_TEMPLATES)


def _toy_variant(family: str, seed: int) -> Tuple[FactorGraph, Candidate, Dict[str, float]]:
    """Resample one instance of a structure_toys pattern with fresh integer
    constants/coefficients. Returns (fixed_graph, gold_candidate, augmented solution).

    Constants are chosen so the augmented system holds exactly at an integer gold
    solution; the fixed graph (u terms dropped) is certified inconsistent.
    """
    # ponytail: local constant-resampler; superseded by marc.data.aux_required when present
    if family not in _TEMPLATES:
        raise ValueError(f"unknown family {family!r} (expected one of {FAMILIES})")
    var_names, base_factors, touched = _TEMPLATES[family]
    rng = random.Random(f"toy_variant:{family}:{seed}")
    for _ in range(200):
        sol = {v: float(rng.randint(-3, 3)) for v in var_names}
        u0 = float(rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)))  # u0 == 0 would leave fixed consistent
        coeffs = {fid: float(rng.choice((-2, -1, 1, 2))) for fid in touched}
        factors, edges = [], []
        for fid, terms in base_factors:
            const = -(sum(c * sol[v] for v, c in terms.items()) + coeffs.get(fid, 0.0) * u0)
            expr = " + ".join(f"({c})*{v}" for v, c in terms.items()) + f" + ({const})"
            factors.append(FactorNode(fid, expr))
            edges.extend(Edge(v, fid, float(c)) for v, c in terms.items())
        fixed = FactorGraph(
            variables=[VariableNode(v) for v in var_names], factors=factors, edges=edges
        )
        if not _is_inconsistent(fixed):
            continue
        # ponytail: keep the gold-augmented system inside the frozen baseline
        # solver's stability region. refine's polish (fixed lr=0.2) diverges when
        # lambda_max(A^T A) >= 10; resample stiff draws so gold_oracle (the
        # positive control) always solves. Ceiling: drop this filter if refine
        # ever gets a line search / adaptive lr.
        rows = [
            [float(terms.get(v, 0)) for v in var_names] + [coeffs.get(fid, 0.0)]
            for fid, terms in base_factors
        ]
        rows.append([0.0] * len(var_names) + [1.0])  # the aux-defining factor
        A = np.asarray(rows)
        if float(np.linalg.eigvalsh(A.T @ A)[-1]) >= 9.0:
            continue
        gold = Candidate(aux_var="u", pin_value=u0, insert_coeffs=coeffs)
        solution = dict(sol)
        solution["u"] = u0
        return fixed, gold, solution
    raise RuntimeError(f"could not sample an inconsistent {family} variant (seed={seed})")


def _pin_from_defining_expression(expression: str, aux_var: str) -> float:
    """v from a defining factor 'u - (v)': the expression at u=0 is -v."""
    expr = sp.sympify(expression)
    return float(-expr.subs(sp.Symbol(aux_var), 0))


def make_dataset(
    source: str,
    n: int,
    seed: int,
    K: int = 4,
    families: Optional[Sequence[str]] = None,
    hard_negatives: bool = True,
) -> List[InventionInstance]:
    """Build ``n`` menu-based invention instances.

    ``source="toys"`` uses the local toy-variant resampler (works on main today).
    ``source="aux_required"`` consumes ``marc.data.aux_required`` (contract C3) when
    present; raises an actionable ImportError otherwise. Deterministic per
    ``(source, n, seed, K, families, hard_negatives)``.
    """
    fams = tuple(families) if families else FAMILIES
    raw: List[Tuple[str, int, FactorGraph, Candidate, Dict[str, float]]] = []
    if source == "toys":
        for i in range(n):
            family = fams[i % len(fams)]
            inst_seed = seed + i
            fixed, gold, solution = _toy_variant(family, inst_seed)
            raw.append((family, inst_seed, fixed, gold, solution))
    elif source == "aux_required":
        try:
            from marc.data import aux_required
        except ImportError as exc:
            raise ImportError(
                "make_dataset(source='aux_required') needs marc/data/aux_required.py "
                "(contract C3: generate_instances(n, seed, patterns)), which is not on "
                "this branch yet. Use source='toys' (the local toy-variant resampler) "
                "or merge the aux_required data unit first."
            ) from exc
        for r in aux_required.generate_instances(n, seed, patterns=list(fams)):
            gold = Candidate(
                aux_var=r.aux_var,
                pin_value=_pin_from_defining_expression(r.defining_expression, r.aux_var),
                insert_coeffs={k: float(v) for k, v in r.insert_coeffs.items()},
            )
            raw.append((r.pattern, r.seed, r.fixed_graph, gold, dict(r.solution)))
    else:
        raise ValueError(f"unknown source {source!r} (expected 'toys' or 'aux_required')")

    out: List[InventionInstance] = []
    for family, inst_seed, fixed, gold, solution in raw:
        rng = random.Random(f"menu:{family}:{inst_seed}:{K}:{hard_negatives}")
        menu, gold_idx = build_menu(fixed, gold, K, rng, hard_negatives=hard_negatives)
        out.append(
            InventionInstance(
                id=f"{source}_{family}_s{inst_seed}",
                family=family,
                seed=inst_seed,
                fixed_graph=fixed,
                candidates=menu,
                gold_idx=gold_idx,
                aux_value=float(solution[gold.aux_var]),
                solution=solution,
            )
        )
    return out


def to_padded(instance: InventionInstance) -> PaddedGraph:
    """Clean state c_0 over 2K slots: candidate j owns var slot 2j + factor slot 2j+1.

    Gold's var slot is VARIABLE carrying ``aux_value``; gold's factor slot is FACTOR
    carrying the pin value; every other slot is ABSENT/0.
    """
    K = len(instance.candidates)
    types = torch.full((2 * K,), ABSENT, dtype=torch.long)
    values = torch.zeros(2 * K, dtype=torch.float32)
    g = instance.gold_idx
    types[2 * g] = int(SlotType.VARIABLE)
    values[2 * g] = instance.aux_value
    types[2 * g + 1] = int(SlotType.FACTOR)
    values[2 * g + 1] = instance.candidates[g].pin_value
    return PaddedGraph(types, values)
