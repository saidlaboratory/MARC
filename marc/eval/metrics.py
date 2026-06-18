"""
Evaluation metrics for MARC (§11, TECHNICAL_GUIDE).

All functions are pure — they take Python lists/sequences of booleans or floats
and return a float. No model, no CAS dependency.

Metric definitions (maps to TECHNICAL_GUIDE §11 table):

  solve_rate(results)
    Fraction of problems the checker accepted.  Implements pass@1.
    results: sequence of bool — True if checker accepted that run.

  generalization_gap(train_results, test_results)
    (in-distribution solve rate) − (held-out-structure solve rate).
    Positive gap → model is over-fitting to training distribution.
    Tests hypothesis H1 (derive-not-recall): MARC should show a smaller gap
    than a CoT baseline at equal scale.

  entrapment_rate(trajectories, tol)
    Fraction of runs that stalled at a fixed point with energy > tol, i.e.
    the solver got "trapped" and never reached E=0.
    trajectories: sequence of float — the final energy E(x) for each run.
    tol: tolerance below which E is considered solved (default 1e-6).
    Compares noise-on vs noise-off to test RQ2: does injected noise reduce
    entrapment?  Call once per condition and compare.

  perturbation_robustness(baseline_results, perturbed_results)
    Solve-rate drop when problem constants are perturbed.
    baseline_results:  bool sequence on original problems.
    perturbed_results: bool sequence on constant-perturbed versions of the same
                       problems (same order).
    Returns baseline_rate − perturbed_rate.  A large drop signals the model is
    recalling memorized constants rather than deriving the solution.
"""

from __future__ import annotations

from typing import Sequence


def solve_rate(results: Sequence[bool]) -> float:
    """Fraction of problems the checker accepted (pass@1)."""
    if len(results) == 0:
        raise ValueError("results must be non-empty")
    return sum(bool(r) for r in results) / len(results)


def generalization_gap(
    train_results: Sequence[bool],
    test_results: Sequence[bool],
) -> float:
    """In-distribution solve rate minus held-out-structure solve rate."""
    return solve_rate(train_results) - solve_rate(test_results)


def entrapment_rate(
    trajectories: Sequence[float],
    tol: float = 1e-6,
) -> float:
    """Fraction of runs whose final energy exceeded tol (trapped at a fixed point)."""
    if len(trajectories) == 0:
        raise ValueError("trajectories must be non-empty")
    return sum(1 for e in trajectories if e > tol) / len(trajectories)


def perturbation_robustness(
    baseline_results: Sequence[bool],
    perturbed_results: Sequence[bool],
) -> float:
    """Solve-rate drop under constant perturbation (baseline − perturbed)."""
    if len(baseline_results) != len(perturbed_results):
        raise ValueError("baseline and perturbed result lists must have equal length")
    return solve_rate(baseline_results) - solve_rate(perturbed_results)
