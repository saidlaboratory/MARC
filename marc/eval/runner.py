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
    """Shift every factor's constant term by delta (changes the problem's solution)."""
    factors = [
        FactorNode(f.id, str(sp.sympify(f.expression) + delta))
        for f in graph.factors
    ]
    return FactorGraph(variables=graph.variables, factors=factors, edges=graph.edges)


def run_eval(
    problems: list[Problem],
    solver: DummySolver | None = None,
    checker: Checker | None = None,
    perturb_delta: float = 0.1,
    n_samples: int = 1,
) -> dict[str, Any]:
    """Run solver+checker on problems and return §11 metrics as a JSON-serialisable dict."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    solver = solver or DummySolver()
    checker = checker or Checker()

    results: list[bool] = []          # pass@1: did the first candidate pass?
    attempts_per_problem: list[list[bool]] = []  # accept/reject of every sample
    max_residuals: list[float] = []
    candidates: list[list[float]] = []
    stages: list[str] = []
    perturbed_results: list[bool] = []

    for p in problems:
        samples = solver.sample(p, n_samples)
        sample_results = [checker.verify(p.graph, x) for x in samples]
        attempts_per_problem.append([r.accepted for r in sample_results])

        first = sample_results[0]
        results.append(first.accepted)
        max_residuals.append(first.max_residual)
        candidates.append(samples[0])
        stages.append(first.stage)

        # does the first answer still solve the problem once its constants shift?
        perturbed = checker.verify(perturb_constants(p.graph, perturb_delta), samples[0])
        perturbed_results.append(perturbed.accepted)

    mid = max(1, len(results) // 2)
    train_results = results[:mid]
    test_results = results[mid:] if len(results) > mid else results[:mid]

    metrics: dict[str, Any] = {
        "n_problems": len(problems),
        "n_samples": n_samples,
        "solve_rate": solve_rate(results),
        "pass_at_k": pass_at_k(attempts_per_problem, n_samples),
        "generalization_gap": generalization_gap(train_results, test_results),
        "entrapment_rate": entrapment_rate(max_residuals),
        "perturbation_robustness": perturbation_robustness(results, perturbed_results),
        "problem_ids": [p.id for p in problems],
        "per_problem": [
            {
                "id": p.id,
                "accepted": r,
                "candidate": [round(v, 6) for v in x],
                "max_residual": mr,
                "reject_stage": st,
            }
            for p, r, x, mr, st in zip(problems, results, candidates, max_residuals, stages)
        ],
    }
    return metrics
