"""Stub eval runner with a DummySolver — exercises the metrics pipeline without a model or CAS."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from marc.eval.metrics import (
    entrapment_rate,
    generalization_gap,
    perturbation_robustness,
    solve_rate,
)


@dataclass
class Problem:
    id: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DummySolver:
    """Returns random solve decisions and energies, seeded for reproducibility."""

    def __init__(
        self,
        solve_prob: float = 0.7,
        entrapment_prob: float = 0.2,
        energy_if_solved: float = 0.0,
        energy_if_trapped: float = 1.5,
        seed: int | None = 42,
    ) -> None:
        self.solve_prob = solve_prob
        self.entrapment_prob = entrapment_prob
        self.energy_if_solved = energy_if_solved
        self.energy_if_trapped = energy_if_trapped
        self._rng = random.Random(seed)

    def solve(self, problem: Problem) -> tuple[bool, float]:
        trapped = self._rng.random() < self.entrapment_prob
        if trapped:
            return False, self.energy_if_trapped
        accepted = self._rng.random() < self.solve_prob
        energy = self.energy_if_solved if accepted else self.energy_if_trapped * 0.5
        return accepted, energy


def run_eval(
    problems: list[Problem],
    solver: DummySolver | None = None,
    perturb_fraction: float = 0.1,
) -> dict[str, Any]:
    """Run solver on problems and return §11 metrics as a JSON-serialisable dict."""
    if solver is None:
        solver = DummySolver()

    results: list[bool] = []
    energies: list[float] = []
    for p in problems:
        accepted, energy = solver.solve(p)
        results.append(accepted)
        energies.append(energy)

    perturbed_solver = DummySolver(
        solve_prob=max(0.0, solver.solve_prob - perturb_fraction),
        entrapment_prob=solver.entrapment_prob,
        seed=99,
    )
    perturbed_results: list[bool] = [perturbed_solver.solve(p)[0] for p in problems]

    mid = max(1, len(results) // 2)
    train_results = results[:mid]
    test_results = results[mid:] if len(results) > mid else results[:mid]

    metrics: dict[str, Any] = {
        "n_problems": len(problems),
        "solve_rate": solve_rate(results),
        "generalization_gap": generalization_gap(train_results, test_results),
        "entrapment_rate": entrapment_rate(energies),
        "perturbation_robustness": perturbation_robustness(results, perturbed_results),
        "problem_ids": [p.id for p in problems],
        "per_problem": [
            {"id": p.id, "accepted": r, "final_energy": e}
            for p, r, e in zip(problems, results, energies)
        ],
    }
    return metrics
