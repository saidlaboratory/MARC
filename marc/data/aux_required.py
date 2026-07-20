"""Procedural aux-required families: H2 structure invention at dataset scale.

Each instance pairs a certified-inconsistent ``fixed_graph`` with a certified
uniquely-solvable ``augmented_graph`` = fixed + one latent ``u`` + its defining
factor, procedurally generalizing the 3 hand-built toys in
``marc/eval/structure_toys.py`` (see its docstring for why the latent must be a
*free* variable with its own defining factor, not an inline expression).

The three patterns generalize the toys' structural signatures — which equations
the latent enters:

* ``offset``  — toy1 shape: base (x, y); u enters eq1, eq2, eq3 (3 eqs + aux).
* ``coupled`` — toy2 shape: base (x, y, z); u enters eq4 only (4 eqs + aux).
* ``shared``  — toy3 shape: base (x, y); u enters eq1 and eq2 (3 eqs + aux).

Constants and coefficients are resampled per seed. Certificates are exact
integer linear algebra: the fixed graph is inconsistent iff
rank(A|b) > rank(A); the augmented graph is uniquely solvable iff
rank(A) == n_vars and the system is consistent; the Checker accepts the gold.
Candidates failing any certificate are rejected and resampled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import sympy as sp

from marc.cas.checker import Checker
from marc.data.templates import _NONZERO, _build_expr
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

AUX_VAR = "u"
_AUX_COEF = [-2, -1, 1, 2]

#: pattern -> base variable ids, base variables per equation ("rows"), and the
#: equation indices the latent u enters ("u_rows"). Shapes mirror the toys.
_PATTERN_SPECS: Dict[str, dict] = {
    "offset": {
        "base": ["x", "y"],
        "rows": [["x", "y"], ["x", "y"], ["x"]],
        "u_rows": (0, 1, 2),
    },
    "coupled": {
        "base": ["x", "y", "z"],
        "rows": [["x", "y", "z"], ["x", "y"], ["y", "z"], ["x"]],
        "u_rows": (3,),
    },
    "shared": {
        "base": ["x", "y"],
        "rows": [["x"], ["y"], ["x", "y"]],
        "u_rows": (0, 1),
    },
}
PATTERNS: List[str] = list(_PATTERN_SPECS)


# --- pinned contract C3 (consumed by the structure policy; do not rename) -----

@dataclass(frozen=True)
class AuxRequiredInstance:
    id: str                          # f"{pattern}_{seed}"
    pattern: str                     # "offset" | "coupled" | "shared"
    seed: int
    fixed_graph: FactorGraph         # inconsistent (certified)
    augmented_graph: FactorGraph     # fixed + aux var + defining factor (certified solvable)
    solution: Dict[str, float]       # augmented gold, keys = augmented var ids (incl. aux)
    aux_var: str                     # "u"
    aux_value: float
    defining_expression: str         # e.g. "u - (2)"
    insert_coeffs: Dict[str, float]  # factor_id -> coefficient of aux_var inserted into that factor


# --- exact linear-algebra certificates ----------------------------------------

def _linear_system(graph: FactorGraph) -> Tuple[sp.Matrix, sp.Matrix, list]:
    """Return (A, b, symbols) with A x = b; raises on nonlinear factor expressions."""
    symbols = [sp.Symbol(v.id) for v in graph.variables]
    exprs = [sp.sympify(f.expression) for f in graph.factors]
    A, b = sp.linear_eq_to_matrix(exprs, symbols)
    return A, b, symbols


def _is_inconsistent(graph: FactorGraph) -> bool:
    """Cheap exact unsolvability certificate: rank(A|b) > rank(A)."""
    A, b, _ = _linear_system(graph)
    return A.row_join(b).rank() > A.rank()


def _is_uniquely_solvable(graph: FactorGraph) -> bool:
    """Consistent with a unique solution: rank(A) == rank(A|b) == n_vars."""
    A, b, symbols = _linear_system(graph)
    r = A.rank()
    return r == len(symbols) and A.row_join(b).rank() == r


def verify_instance(inst: AuxRequiredInstance) -> bool:
    """Re-run every certificate; shared by the generation rejection loop and tests."""
    if not _is_inconsistent(inst.fixed_graph):
        return False
    if not _is_uniquely_solvable(inst.augmented_graph):
        return False
    gold = [inst.solution[v.id] for v in inst.augmented_graph.variables]
    return Checker().verify(inst.augmented_graph, gold).accepted


# --- generation ---------------------------------------------------------------

@dataclass
class AuxRequiredTemplate:
    """One aux-required pattern as a ProblemGenerator-compatible template.

    ``generate`` returns the SOLVABLE (augmented) side so ProblemGenerator's
    CAS-accept invariant holds; ``generate_instance`` returns the full paired
    AuxRequiredInstance (contract C3).
    """

    pattern: str
    name: str = ""

    def __post_init__(self) -> None:
        if self.pattern not in _PATTERN_SPECS:
            raise ValueError(f"unknown pattern {self.pattern!r}; expected one of {PATTERNS}")
        if not self.name:
            self.name = f"AuxRequired_{self.pattern}"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        inst = self.generate_instance(seed)
        return inst.augmented_graph, dict(inst.solution)

    def generate_instance(self, seed: int = None) -> AuxRequiredInstance:
        spec = _PATTERN_SPECS[self.pattern]
        rng = random.Random(seed)
        for _ in range(200):
            inst = self._sample(spec, rng, seed)
            if verify_instance(inst):
                return inst
        raise RuntimeError(f"could not generate a certified {self.name} instance (seed={seed})")

    def _sample(self, spec: dict, rng: random.Random, seed: int) -> AuxRequiredInstance:
        base = spec["base"]
        gold = {v: rng.randint(-4, 4) for v in base}
        u_star = rng.choice(_NONZERO)  # nonzero so dropping u matters
        full_gold = {**gold, AUX_VAR: u_star}

        aug_factors, fix_factors = [], []
        aug_edges, fix_edges = [], []
        insert_coeffs: Dict[str, float] = {}
        for i, row_vars in enumerate(spec["rows"]):
            fid = f"eq{i + 1}"
            names = list(row_vars)
            coeffs = [rng.choice(_NONZERO) for _ in names]
            if i in spec["u_rows"]:
                names.append(AUX_VAR)
                coeffs.append(rng.choice(_AUX_COEF))
                insert_coeffs[fid] = float(coeffs[-1])
            rhs = sum(c * full_gold[v] for c, v in zip(coeffs, names))
            aug_factors.append(FactorNode(fid, _build_expr(coeffs, names, rhs)))
            aug_edges += [Edge(v, fid, float(c)) for c, v in zip(coeffs, names)]
            # fixed graph: same equation with the u term deleted, same RHS
            # (exactly the toys' construction — the deleted latent's contribution
            # is what tips the system into contradiction).
            n_base = len(row_vars)
            fix_factors.append(
                FactorNode(fid, _build_expr(coeffs[:n_base], names[:n_base], rhs))
            )
            fix_edges += [
                Edge(v, fid, float(c)) for c, v in zip(coeffs[:n_base], names[:n_base])
            ]

        defining = f"{AUX_VAR} - ({u_star})"
        aug_factors.append(FactorNode("aux", defining))
        aug_edges.append(Edge(AUX_VAR, "aux", 1.0))

        augmented = FactorGraph(
            variables=[VariableNode(v, 0.0) for v in base] + [VariableNode(AUX_VAR, 0.0)],
            factors=aug_factors,
            edges=aug_edges,
        )
        fixed = FactorGraph(
            variables=[VariableNode(v, 0.0) for v in base],
            factors=fix_factors,
            edges=fix_edges,
        )
        solution = {v: float(gold[v]) for v in base}
        solution[AUX_VAR] = float(u_star)
        return AuxRequiredInstance(
            id=f"{self.pattern}_{seed}",
            pattern=self.pattern,
            seed=seed,
            fixed_graph=fixed,
            augmented_graph=augmented,
            solution=solution,
            aux_var=AUX_VAR,
            aux_value=float(u_star),
            defining_expression=defining,
            insert_coeffs=insert_coeffs,
        )


def generate_instances(
    n: int, seed: int = 0, patterns: Sequence[str] | None = None
) -> List[AuxRequiredInstance]:
    """n instances round-robin over patterns; instance i uses seed seed + i."""
    pats = list(patterns) if patterns is not None else PATTERNS
    return [
        AuxRequiredTemplate(pats[i % len(pats)]).generate_instance(seed + i)
        for i in range(n)
    ]


#: The procedural aux-required template family (one template per pattern).
AUX_REQUIRED_TEMPLATES = [AuxRequiredTemplate(p) for p in PATTERNS]
