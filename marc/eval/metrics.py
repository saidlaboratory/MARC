"""
Evaluation metrics for MARC (maps to TECHNICAL_GUIDE §11).

Pure functions over sequences of bool/float — no model, no CAS dependency.

  solve_rate            — pass@1: fraction the checker accepted.
  pass_at_k             — pass@k: fraction solved within k attempts.
  generalization_gap    — in-distribution rate − held-out-structure rate (H1).
  entrapment_rate       — fraction of runs stalled at energy > tol (RQ2).
  entrapment_reduction  — entrapment without noise − with noise (RQ2 ablation).
  perturbation_robustness — solve-rate drop when constants are perturbed.
  derivation_verifiability — fraction of accepted solutions that formally verify.
  wilson_interval        — 95% Wilson score CI for a binomial solve rate.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple


def solve_rate(results: Sequence[bool]) -> float:
    """Fraction of problems the checker accepted (pass@1)."""
    if len(results) == 0:
        raise ValueError("results must be non-empty")
    return sum(bool(r) for r in results) / len(results)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k successes in n trials.

    Preferred over the normal approximation for small n and rates near 0/1 (exactly
    our regime). Returns (low, high), both clamped to [0, 1)."""
    if n == 0:
        raise ValueError("n must be positive")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pass_at_k(results_per_problem: Sequence[Sequence[bool]], k: int) -> float:
    """Fraction of problems with at least one accepted attempt among the first k (pass@k)."""
    if len(results_per_problem) == 0:
        raise ValueError("results_per_problem must be non-empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    solved = sum(
        1 for attempts in results_per_problem if any(bool(a) for a in attempts[:k])
    )
    return solved / len(results_per_problem)


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


def entrapment_reduction(
    noise_off_trajectories: Sequence[float],
    noise_on_trajectories: Sequence[float],
    tol: float = 1e-6,
) -> float:
    """Entrapment rate without noise minus with noise (positive ⇒ noise helps; RQ2)."""
    return entrapment_rate(noise_off_trajectories, tol) - entrapment_rate(
        noise_on_trajectories, tol
    )


def perturbation_robustness(
    baseline_results: Sequence[bool],
    perturbed_results: Sequence[bool],
) -> float:
    """Solve-rate drop under constant perturbation (baseline − perturbed)."""
    if len(baseline_results) != len(perturbed_results):
        raise ValueError("baseline and perturbed result lists must have equal length")
    return solve_rate(baseline_results) - solve_rate(perturbed_results)


def derivation_verifiability(
    accepted_solutions: Sequence,
    verify_fn: Callable,
) -> float:
    """Fraction of checker-accepted solutions that pass a stronger formal verifier."""
    if not accepted_solutions:
        return 0.0
    return sum(1 for s in accepted_solutions if verify_fn(s)) / len(accepted_solutions)
