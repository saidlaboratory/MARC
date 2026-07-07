"""H2 evaluation: does inventing an auxiliary object help solving? (CONCEPT.md H2,
TECHNICAL_GUIDE §10, §11 "Intermediate-object usage").

Full discrete structure diffusion (D3PM over slot types, :mod:`marc.model.structure_head`)
is [Frontier] and not yet trained end-to-end. Per TECHNICAL_GUIDE §14 ("build the
simplest thing that exhibits the phenomenon first"), this module's **structure
model** is the MVP stand-in: energy-guided best-of-N search over two structural
hypotheses per restart — the *fixed* graph (no auxiliary slot) and an *augmented*
graph with one extra auxiliary variable + its defining factor(s) turned on. This
mirrors :class:`marc.model.structure_head.StructureHead`'s ABSENT-vs-active slot
distinction, using the exact CAS energy in place of a trained categorical policy.

Three toy families (each a genuine textbook auxiliary-quantity trick, matching
CONCEPT.md H2's "lemmas and auxiliary quantities"):

* ``sum_product``      — x+y=s, xy=p (bilinear). Aux ``d=x-y`` turns the coupled
  bilinear pair into one 1-D quadratic in ``d`` plus a linear link back to x, y.
* ``bilinear_product``  — x*y=A, y*z=B, x*z=C (3-var bilinear). Aux ``w=x*y*z``
  gives ``w**2 = A*B*C``, then x, y, z each follow by division.
* ``quadratic_link``    — x**2+y=a, x**2-y=b. Aux ``z=x**2`` linearises both
  equations in (z, y); only the definition ``z=x**2`` stays nonlinear.

The auxiliary variable is always *added*, never removed — the fixed graph's factors
stay intact in the augmented graph, so solving the augmented graph never changes
the original solution set. What we measure is whether the extra structure helps the
*search* find it, and whether the winning run actually used the auxiliary slot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from marc.cas.checker import Checker
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.refine.iterative import refine

_NONZERO = [-4, -3, -2, -1, 1, 2, 3, 4]


# --------------------------------------------------------------------------- toys

@dataclass
class ToyProblem:
    name: str
    seed: int
    fixed_graph: FactorGraph
    aux_graph: FactorGraph  # fixed_graph's variables/factors + one aux slot
    aux_var: str
    solution: Dict[str, float]  # one known assignment for the *base* variables
    description: str


def toy_sum_product(seed: int = 0) -> ToyProblem:
    """x + y = s,  x*y = p.  Aux: d = x - y,  d**2 = s**2 - 4p (Vieta's difference)."""
    rng = random.Random(seed)
    x_star = rng.choice(_NONZERO)
    y_star = rng.choice(_NONZERO)
    while y_star == x_star:  # avoid the degenerate d=0 instance
        y_star = rng.choice(_NONZERO)
    s, p = x_star + y_star, x_star * y_star

    variables = [VariableNode("x"), VariableNode("y")]
    factors = [
        FactorNode("eq1", f"x + y - ({s})"),
        FactorNode("eq2", f"x*y - ({p})"),
    ]
    edges = [Edge("x", "eq1", 1), Edge("y", "eq1", 1), Edge("x", "eq2", 1), Edge("y", "eq2", 1)]
    fixed = FactorGraph(variables=variables, factors=factors, edges=edges)

    aux_vars = variables + [VariableNode("d")]
    aux_factors = factors + [
        FactorNode("eq_def", "d - (x - y)"),
        FactorNode("eq_aux", f"d**2 - (({s})**2 - 4*({p}))"),
    ]
    aux_edges = edges + [
        Edge("x", "eq_def", 1), Edge("y", "eq_def", -1), Edge("d", "eq_def", 1),
        Edge("d", "eq_aux", 1),
    ]
    aux = FactorGraph(variables=aux_vars, factors=aux_factors, edges=aux_edges)

    return ToyProblem(
        "sum_product", seed, fixed, aux, "d",
        {"x": float(x_star), "y": float(y_star)},
        f"x+y={s}, x*y={p} (solution x={x_star}, y={y_star})",
    )


def toy_bilinear_product(seed: int = 0) -> ToyProblem:
    """x*y=A, y*z=B, x*z=C.  Aux: w = x*y*z,  w**2 = A*B*C, then x=w/B, y=w/C, z=w/A."""
    rng = random.Random(seed)
    x_star, y_star, z_star = (rng.choice(_NONZERO) for _ in range(3))
    A, B, C = x_star * y_star, y_star * z_star, x_star * z_star

    variables = [VariableNode("x"), VariableNode("y"), VariableNode("z")]
    factors = [
        FactorNode("eq_xy", f"x*y - ({A})"),
        FactorNode("eq_yz", f"y*z - ({B})"),
        FactorNode("eq_xz", f"x*z - ({C})"),
    ]
    edges = [
        Edge("x", "eq_xy", 1), Edge("y", "eq_xy", 1),
        Edge("y", "eq_yz", 1), Edge("z", "eq_yz", 1),
        Edge("x", "eq_xz", 1), Edge("z", "eq_xz", 1),
    ]
    fixed = FactorGraph(variables=variables, factors=factors, edges=edges)

    aux_vars = variables + [VariableNode("w")]
    aux_factors = factors + [
        FactorNode("eq_defw", "w - (x*y*z)"),
        FactorNode("eq_auxw", f"w**2 - (({A})*({B})*({C}))"),
    ]
    aux_edges = edges + [
        Edge("x", "eq_defw", 1), Edge("y", "eq_defw", 1), Edge("z", "eq_defw", 1),
        Edge("w", "eq_defw", -1), Edge("w", "eq_auxw", 1),
    ]
    aux = FactorGraph(variables=aux_vars, factors=aux_factors, edges=aux_edges)

    return ToyProblem(
        "bilinear_product", seed, fixed, aux, "w",
        {"x": float(x_star), "y": float(y_star), "z": float(z_star)},
        f"x*y={A}, y*z={B}, x*z={C} (solution x={x_star}, y={y_star}, z={z_star})",
    )


def toy_quadratic_link(seed: int = 0) -> ToyProblem:
    """x**2 + y = a,  x**2 - y = b.  Aux: z = x**2 linearises both in (z, y)."""
    rng = random.Random(seed)
    x_star = rng.choice(_NONZERO)
    y_star = rng.randint(-6, 6)
    a, b = x_star ** 2 + y_star, x_star ** 2 - y_star

    variables = [VariableNode("x"), VariableNode("y")]
    factors = [
        FactorNode("eq1", f"x**2 + y - ({a})"),
        FactorNode("eq2", f"x**2 - y - ({b})"),
    ]
    edges = [Edge("x", "eq1", 1), Edge("y", "eq1", 1), Edge("x", "eq2", 1), Edge("y", "eq2", -1)]
    fixed = FactorGraph(variables=variables, factors=factors, edges=edges)

    aux_vars = variables + [VariableNode("z")]
    aux_factors = factors + [
        FactorNode("eq_defz", "z - x**2"),
        FactorNode("eq_z1", f"z + y - ({a})"),
        FactorNode("eq_z2", f"z - y - ({b})"),
    ]
    aux_edges = edges + [
        Edge("x", "eq_defz", 1), Edge("z", "eq_defz", -1),
        Edge("z", "eq_z1", 1), Edge("y", "eq_z1", 1),
        Edge("z", "eq_z2", 1), Edge("y", "eq_z2", -1),
    ]
    aux = FactorGraph(variables=aux_vars, factors=aux_factors, edges=aux_edges)

    return ToyProblem(
        "quadratic_link", seed, fixed, aux, "z",
        {"x": float(x_star), "y": float(y_star)},
        f"x**2+y={a}, x**2-y={b} (solution x={x_star}, y={y_star})",
    )


TOYS: Dict[str, Callable[[int], ToyProblem]] = {
    "sum_product": toy_sum_product,
    "bilinear_product": toy_bilinear_product,
    "quadratic_link": toy_quadratic_link,
}


# ------------------------------------------------------------------------ solving

def _random_init(n: int, scale: float, rng: np.random.Generator) -> List[float]:
    return (rng.standard_normal(n) * scale).tolist()


# The toy energies here are bilinear/quartic (not the near-quadratic bowls the P1/P2
# linear-system suites use), so the default polish (400 steps @ lr=0.2) undershoots
# the checker's 1e-9 symbolic snap tolerance even when the exploration phase has
# found the right basin. A longer, gentler polish closes that gap without changing
# which basin is found (see :func:`marc.refine.iterative.refine`'s polish docstring).
_POLISH_STEPS = 2500
_POLISH_LR = 0.1


def _refine(graph: FactorGraph, x0: List[float], *, steps: int, seed: int):
    return refine(
        graph, x0, steps=steps, seed=seed,
        polish_steps=_POLISH_STEPS, polish_lr=_POLISH_LR,
    )


@dataclass
class RunRecord:
    """One structure-model restart's outcome, projected onto the base variables."""

    toy: str
    seed: int
    accepted: bool
    used_aux: bool
    x_base: List[float]
    energy: float
    description: str = ""

    def to_dict(self, *, model: str = "") -> dict:
        return {
            "toy": self.toy,
            "seed": self.seed,
            "model": model,
            "description": self.description,
            "accepted": self.accepted,
            "used_aux": self.used_aux,
            "x_base": [round(v, 8) for v in self.x_base],
            "energy": self.energy,
        }


def solve_fixed(problem: ToyProblem, k: int, *, steps: int = 300, seed: int = 0) -> RunRecord:
    """The *fixed* model: k restarts on the fixed graph only — no auxiliary slot exists."""
    rng = np.random.default_rng(seed)
    checker = Checker()
    n = len(problem.fixed_graph.variables)
    best: RunRecord | None = None
    for _ in range(k):
        trace = _refine(
            problem.fixed_graph, _random_init(n, 3.0, rng), steps=steps,
            seed=int(rng.integers(0, 2 ** 31 - 1)),
        )
        result = checker.verify(problem.fixed_graph, trace.x)
        record = RunRecord(
            problem.name, problem.seed, result.accepted, False, trace.x,
            trace.best_energy, problem.description,
        )
        if result.accepted:
            return record
        if best is None or trace.best_energy < best.energy:
            best = record
    return best


def solve_structure(problem: ToyProblem, k: int, *, steps: int = 300, seed: int = 0) -> RunRecord:
    """The *structure* model: k restarts split between aux-ABSENT and aux-ACTIVE graphs;
    keep the lowest-energy accepted candidate (energy-guided stand-in for a trained
    slot-activation policy, see module docstring)."""
    rng = np.random.default_rng(seed)
    checker = Checker()
    n_base = len(problem.fixed_graph.variables)
    candidates: List[RunRecord] = []
    for i in range(k):
        use_aux = i % 2 == 1
        graph = problem.aux_graph if use_aux else problem.fixed_graph
        n = len(graph.variables)
        trace = _refine(graph, _random_init(n, 3.0, rng), steps=steps, seed=int(rng.integers(0, 2 ** 31 - 1)))
        x_base = trace.x[:n_base]
        result = checker.verify(problem.fixed_graph, x_base)
        candidates.append(RunRecord(
            problem.name, problem.seed, result.accepted, use_aux, x_base,
            trace.best_energy, problem.description,
        ))

    solved = [c for c in candidates if c.accepted]
    if solved:
        return min(solved, key=lambda c: c.energy)
    return min(candidates, key=lambda c: c.energy)


# ------------------------------------------------------------------------ metrics

def auxiliary_usage_rate(records: Sequence[RunRecord]) -> float:
    """Fraction of *solved* structure-model runs whose winning restart used the
    auxiliary slot (H2 / TECHNICAL_GUIDE §11 "Intermediate-object usage").

    Undefined (returns 0.0) if nothing solved — usage only means something once the
    model actually reaches a checker-accepted state.
    """
    solved = [r for r in records if r.accepted]
    if not solved:
        return 0.0
    return sum(1 for r in solved if r.used_aux) / len(solved)


def solve_rate(records: Sequence[RunRecord]) -> float:
    if not records:
        raise ValueError("records must be non-empty")
    return sum(1 for r in records if r.accepted) / len(records)


# -------------------------------------------------------------------------- suite

def run_h2_suite(
    toy_names: Sequence[str] | None = None,
    *,
    n_instances: int = 20,
    k: int = 8,
    steps: int = 300,
    base_seed: int = 0,
) -> Tuple[dict, List[RunRecord], List[RunRecord]]:
    """Run fixed vs. structure model over ``n_instances`` random instances per toy.

    Returns (summary_dict, fixed_records, structure_records).
    """
    toy_names = list(toy_names or TOYS.keys())
    fixed_records: List[RunRecord] = []
    structure_records: List[RunRecord] = []
    per_toy: Dict[str, dict] = {}

    for name in toy_names:
        make = TOYS[name]
        toy_fixed: List[RunRecord] = []
        toy_structure: List[RunRecord] = []
        for i in range(n_instances):
            seed = base_seed + i
            problem = make(seed)
            toy_fixed.append(solve_fixed(problem, k, steps=steps, seed=seed * 1009 + 1))
            toy_structure.append(solve_structure(problem, k, steps=steps, seed=seed * 1009 + 2))

        fixed_records.extend(toy_fixed)
        structure_records.extend(toy_structure)
        per_toy[name] = {
            "n_instances": n_instances,
            "fixed_solve_rate": solve_rate(toy_fixed),
            "structure_solve_rate": solve_rate(toy_structure),
            "auxiliary_usage_rate": auxiliary_usage_rate(toy_structure),
            "solve_rate_gain": solve_rate(toy_structure) - solve_rate(toy_fixed),
        }

    summary = {
        "k": k,
        "n_instances_per_toy": n_instances,
        "toys": per_toy,
        "overall_fixed_solve_rate": solve_rate(fixed_records),
        "overall_structure_solve_rate": solve_rate(structure_records),
        "overall_auxiliary_usage_rate": auxiliary_usage_rate(structure_records),
    }
    return summary, fixed_records, structure_records
