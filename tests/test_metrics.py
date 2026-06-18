"""Tests for marc/eval/metrics.py — all inputs are hand-made arrays."""

import math
import pytest

from marc.eval.metrics import (
    entrapment_rate,
    generalization_gap,
    perturbation_robustness,
    solve_rate,
)


# ---------------------------------------------------------------------------
# solve_rate
# ---------------------------------------------------------------------------

class TestSolveRate:
    def test_eight_of_ten(self):
        results = [True] * 8 + [False] * 2
        assert solve_rate(results) == pytest.approx(0.8)

    def test_all_solved(self):
        assert solve_rate([True, True, True]) == pytest.approx(1.0)

    def test_none_solved(self):
        assert solve_rate([False, False]) == pytest.approx(0.0)

    def test_single_solved(self):
        assert solve_rate([True]) == pytest.approx(1.0)

    def test_single_unsolved(self):
        assert solve_rate([False]) == pytest.approx(0.0)

    def test_mixed_five(self):
        # 3 out of 5
        assert solve_rate([True, False, True, False, True]) == pytest.approx(0.6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            solve_rate([])

    def test_accepts_integers_as_bool(self):
        # 1 and 0 should be treated as True/False
        assert solve_rate([1, 0, 1]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# generalization_gap
# ---------------------------------------------------------------------------

class TestGeneralizationGap:
    def test_positive_gap(self):
        # train: 9/10 = 0.9, test: 5/10 = 0.5 → gap = 0.4
        train = [True] * 9 + [False]
        test = [True] * 5 + [False] * 5
        assert generalization_gap(train, test) == pytest.approx(0.4)

    def test_zero_gap(self):
        same = [True, False, True, False]
        assert generalization_gap(same, same) == pytest.approx(0.0)

    def test_negative_gap(self):
        # test set happens to solve more than train (unusual but valid)
        train = [False, False]
        test = [True, True]
        assert generalization_gap(train, test) == pytest.approx(-1.0)

    def test_single_item_each(self):
        assert generalization_gap([True], [False]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# entrapment_rate
# ---------------------------------------------------------------------------

class TestEntrapmentRate:
    def test_two_of_five_trapped(self):
        # energies above tol=1e-6 are "trapped"
        energies = [0.0, 0.0, 0.0, 1.5, 2.0]
        assert entrapment_rate(energies) == pytest.approx(0.4)

    def test_none_trapped(self):
        energies = [0.0, 1e-10, 5e-9]
        assert entrapment_rate(energies) == pytest.approx(0.0)

    def test_all_trapped(self):
        energies = [1.0, 2.0, 3.0]
        assert entrapment_rate(energies) == pytest.approx(1.0)

    def test_custom_tolerance(self):
        energies = [0.05, 0.1, 0.001, 0.0]
        # tol=0.09 → only 0.1 > 0.09, so 1/4 = 0.25
        assert entrapment_rate(energies, tol=0.09) == pytest.approx(0.25)

    def test_exactly_at_tolerance_not_trapped(self):
        # value equal to tol is NOT > tol → not trapped
        energies = [1e-6]
        assert entrapment_rate(energies, tol=1e-6) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            entrapment_rate([])


# ---------------------------------------------------------------------------
# perturbation_robustness
# ---------------------------------------------------------------------------

class TestPerturbationRobustness:
    def test_typical_drop(self):
        # baseline: 8/10, perturbed: 5/10 → robustness = 0.3
        baseline = [True] * 8 + [False] * 2
        perturbed = [True] * 5 + [False] * 5
        assert perturbation_robustness(baseline, perturbed) == pytest.approx(0.3)

    def test_no_drop(self):
        results = [True, False, True]
        assert perturbation_robustness(results, results) == pytest.approx(0.0)

    def test_full_collapse(self):
        baseline = [True, True, True, True]
        perturbed = [False, False, False, False]
        assert perturbation_robustness(baseline, perturbed) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            perturbation_robustness([True, True], [False])

    def test_single_pair(self):
        assert perturbation_robustness([True], [False]) == pytest.approx(1.0)
        assert perturbation_robustness([False], [True]) == pytest.approx(-1.0)
