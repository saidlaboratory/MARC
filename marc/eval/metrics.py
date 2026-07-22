"""
Evaluation metrics for MARC (maps to TECHNICAL_GUIDE §11).

Pure functions over sequences of bool/float — no model, no CAS dependency.

  solve_rate            — pass@1: fraction the checker accepted.
  pass_at_k             — pass@k: fraction solved within k attempts.
  generalization_gap    — in-distribution rate − held-out-structure rate (H1).
  entrapment_rate       — fraction of runs stalled at energy > tol (RQ2).
  entrapment_reduction  — entrapment without noise − with noise (RQ2 ablation).
  perturbation_robustness — solve-rate drop when constants are perturbed.
  wilson_interval        — 95% Wilson score CI for a binomial solve rate.
  restart_budget_curve   — solve rate vs. restart budget k (pass@1..k with CIs).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


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


def rate_cell(k: int, n: int) -> dict:
    """The citable {k, n, rate, ci95} results-JSON cell; vacuous CI when n == 0."""
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci95": [0.0, 1.0]}
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test for p1 > p2 (pooled). Returns (z, one_sided_p).

    Use this for solve-rate comparisons rather than checking whether two 95% CIs
    overlap — non-overlapping 95% CIs is a much stricter bar (~p<0.006) and can miss
    genuine differences. p1 - p2 significant at 0.05 one-sided when z > 1.645."""
    if n1 == 0 or n2 == 0:
        raise ValueError("n1, n2 must be positive")
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 0.5
    z = (p1 - p2) / se
    # one-sided normal tail via erfc
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return z, p


def restart_budget_curve(
    first_success_indices: Sequence[Optional[int]],
    k_max: int,
) -> List[dict]:
    """Solve rate as a function of restart budget, from which-restart-succeeded.

    ``first_success_indices[i]`` is the 0-based index of the first accepted
    candidate for problem i (None if none succeeded). Returns
    ``[{"k": j, "n": n, "rate": r, "ci95": [lo, hi]}]`` for j = 1..k_max, where
    rate is the fraction of problems solved within the first j restarts and the
    CI is the Wilson interval — the pass@1..K curve for free."""
    n = len(first_success_indices)
    if n == 0:
        raise ValueError("first_success_indices must be non-empty")
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    curve: List[dict] = []
    for j in range(1, k_max + 1):
        solved = sum(1 for i in first_success_indices if i is not None and i < j)
        lo, hi = wilson_interval(solved, n)
        curve.append({"k": j, "n": n, "rate": solved / n, "ci95": [lo, hi]})
    return curve


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
