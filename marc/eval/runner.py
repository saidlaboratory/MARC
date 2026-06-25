"""Eval runner: a DummySolver proposes assignments, the Checker decides accept/reject."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from marc.cas.checker import Checker
from marc.graph.graph import FactorGraph
from marc.graph.schema import FactorNode
from marc.eval.metrics import (
    entrapment_rate,
    generalization_gap,
    pass_at_k,
    perturbation_robustness,
    solve_rate,
)


@dataclass
class Problem:
    id: str
    graph: FactorGraph
    solution: list[float]  # known true assignment
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DummySolver:
    """Proposes a candidate assignment per problem; the Checker decides acceptance."""

    def __init__(
        self,
        solve_prob: float = 0.7,
        perturb_scale: float = 0.5,
        seed: int | None = 42,
    ) -> None:
        self.solve_prob = solve_prob
        self.perturb_scale = perturb_scale
        self._rng = random.Random(seed)

    def solve(self, problem: Problem) -> list[float]:
        if self._rng.random() < self.solve_prob:
            return list(problem.solution)
        return [
            v + self._rng.uniform(-1.0, 1.0) * self.perturb_scale
            for v in problem.solution
        ]

    def sample(self, problem: Problem, k: int) -> list[list[float]]:
        """Draw k candidate assignments (the diffusion model would emit several)."""
        return [self.solve(problem) for _ in range(k)]


def perturb_constants(graph: FactorGraph, delta: float) -> FactorGraph:
    """Shift every factor's constant term by delta (changes the problem's solution).

    ``delta`` is shifted in as an *exact* rational (via its decimal string) so the
    perturbed problem keeps rational solutions — otherwise the binary-float value of
    e.g. 0.1 leaves the symbolic checker unable to accept any re-derived answer.
    """
    delta_exact = sp.Rational(str(delta))
    factors = [
        FactorNode(f.id, str(sp.sympify(f.expression) + delta_exact))
        for f in graph.factors
    ]
    return FactorGraph(variables=graph.variables, factors=factors, edges=graph.edges)


@dataclass
class _SplitRun:
    """Raw per-problem outcomes for one set of problems (one split)."""

    problems: list[Problem]
    results: list[bool]               # pass@1 per problem
    attempts: list[list[bool]]        # accept/reject of every sample
    max_residuals: list[float]
    candidates: list[list[float]]
    stages: list[str]
    perturbed_results: list[bool]

    def per_problem(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.id,
                "split": p.metadata.get("split", ""),
                "accepted": r,
                "candidate": [round(v, 6) for v in x],
                "max_residual": mr,
                "reject_stage": st,
                "perturbed_accepted": pr,
            }
            for p, r, x, mr, st, pr in zip(
                self.problems, self.results, self.candidates,
                self.max_residuals, self.stages, self.perturbed_results,
            )
        ]


def _evaluate_split(
    problems: list[Problem],
    solver: Any,
    checker: Checker,
    perturb_delta: float,
    n_samples: int,
    resolve_perturbed: bool = False,
) -> _SplitRun:
    """Run solver+checker over one set of problems, collecting raw outcomes.

    ``resolve_perturbed`` controls the perturbation probe (§11 recall detector):
    when False (legacy ``run_eval``) the *original* answer is re-checked against the
    shifted constants — a stability test. When True (``run_split_eval``) the solver
    is **re-run** on the perturbed problem, so a deriving solver still solves it
    while a memorizing one fails — the informative version.
    """
    results: list[bool] = []
    attempts: list[list[bool]] = []
    max_residuals: list[float] = []
    candidates: list[list[float]] = []
    stages: list[str] = []
    perturbed_results: list[bool] = []

    for p in problems:
        samples = solver.sample(p, n_samples)
        sample_results = [checker.verify(p.graph, x) for x in samples]
        attempts.append([r.accepted for r in sample_results])

        first = sample_results[0]
        results.append(first.accepted)
        max_residuals.append(first.max_residual)
        candidates.append(samples[0])
        stages.append(first.stage)

        perturbed_graph = perturb_constants(p.graph, perturb_delta)
        if resolve_perturbed:
            # re-derive on the shifted problem and check the fresh answer
            perturbed_problem = Problem(
                id=p.id, graph=perturbed_graph, solution=p.solution,
                description=p.description, metadata=p.metadata,
            )
            perturbed_x = solver.sample(perturbed_problem, 1)[0]
            perturbed = checker.verify(perturbed_graph, perturbed_x)
        else:
            # does the original answer still hold once its constants shift?
            perturbed = checker.verify(perturbed_graph, samples[0])
        perturbed_results.append(perturbed.accepted)

    return _SplitRun(
        problems, results, attempts, max_residuals,
        candidates, stages, perturbed_results,
    )


def run_eval(
    problems: list[Problem],
    solver: Any | None = None,
    checker: Checker | None = None,
    perturb_delta: float = 0.1,
    n_samples: int = 1,
) -> dict[str, Any]:
    """Run solver+checker on problems and return §11 metrics as a JSON-serialisable dict.

    The generalization gap here is a naive first-half/second-half split — for a real
    structural split (in-distribution vs. held-out structure) use
    :func:`run_split_eval`.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    solver = solver or DummySolver()
    checker = checker or Checker()

    run = _evaluate_split(problems, solver, checker, perturb_delta, n_samples)
    results = run.results

    mid = max(1, len(results) // 2)
    train_results = results[:mid]
    test_results = results[mid:] if len(results) > mid else results[:mid]

    return {
        "n_problems": len(problems),
        "n_samples": n_samples,
        "solve_rate": solve_rate(results),
        "pass_at_k": pass_at_k(run.attempts, n_samples),
        "generalization_gap": generalization_gap(train_results, test_results),
        "entrapment_rate": entrapment_rate(run.max_residuals),
        "perturbation_robustness": perturbation_robustness(results, run.perturbed_results),
        "problem_ids": [p.id for p in problems],
        "per_problem": [
            {k: pp[k] for k in ("id", "accepted", "candidate", "max_residual", "reject_stage")}
            for pp in run.per_problem()
        ],
    }


def _split_summary(run: _SplitRun, n_samples: int) -> dict[str, Any]:
    """Per-split capability metrics."""
    return {
        "n_problems": len(run.problems),
        "solve_rate": solve_rate(run.results),
        "pass_at_k": pass_at_k(run.attempts, n_samples),
        "perturbation_robustness": perturbation_robustness(run.results, run.perturbed_results),
    }


def run_split_eval(
    in_distribution: list[Problem],
    held_out: list[Problem],
    solver: Any | None = None,
    checker: Checker | None = None,
    perturb_delta: float = 0.1,
    n_samples: int = 1,
    solver_name: str = "",
) -> dict[str, Any]:
    """Evaluate explicit in-distribution and held-out-structure splits (§11, H1).

    Computes the generalization gap from the two *real* split solve rates rather
    than an arbitrary half-split, and reports per-split capability metrics plus a
    flat per-problem record across both splits.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not in_distribution or not held_out:
        raise ValueError("both splits must be non-empty")
    solver = solver or DummySolver()
    checker = checker or Checker()

    id_run = _evaluate_split(
        in_distribution, solver, checker, perturb_delta, n_samples, resolve_perturbed=True
    )
    ho_run = _evaluate_split(
        held_out, solver, checker, perturb_delta, n_samples, resolve_perturbed=True
    )

    id_summary = _split_summary(id_run, n_samples)
    ho_summary = _split_summary(ho_run, n_samples)
    gap = generalization_gap(id_run.results, ho_run.results)

    return {
        "solver": solver_name or getattr(solver, "name", type(solver).__name__),
        "n_samples": n_samples,
        "perturb_delta": perturb_delta,
        "splits": {
            "in_distribution": id_summary,
            "held_out_structure": ho_summary,
        },
        "generalization_gap": gap,
        "overall_solve_rate": solve_rate(id_run.results + ho_run.results),
        "per_problem": id_run.per_problem() + ho_run.per_problem(),
    }
