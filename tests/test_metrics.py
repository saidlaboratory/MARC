"""Tests for marc/eval/metrics.py — all inputs are hand-made arrays."""

import pytest

from marc.eval.metrics import (
    derivation_verifiability,
    entrapment_rate,
    entrapment_reduction,
    generalization_gap,
    pass_at_k,
    perturbation_robustness,
    restart_budget_curve,
    solve_rate,
    two_proportion_z,
    wilson_interval,
)


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


class TestPassAtK:
    def test_k1_uses_first_attempt(self):
        # first attempts: True, False, True → 2/3
        per_problem = [[True, True], [False, True], [True, False]]
        assert pass_at_k(per_problem, k=1) == pytest.approx(2 / 3)

    def test_k2_finds_later_success(self):
        # problem 2 fails first but succeeds on second attempt → all 3 solved
        per_problem = [[True, True], [False, True], [True, False]]
        assert pass_at_k(per_problem, k=2) == pytest.approx(1.0)

    def test_k_larger_than_attempts(self):
        # k beyond available attempts just uses all of them
        per_problem = [[False, False], [False, True]]
        assert pass_at_k(per_problem, k=10) == pytest.approx(0.5)

    def test_none_solved(self):
        per_problem = [[False, False], [False]]
        assert pass_at_k(per_problem, k=2) == pytest.approx(0.0)

    def test_k1_matches_solve_rate_of_first_attempts(self):
        per_problem = [[True], [False], [True], [True]]
        assert pass_at_k(per_problem, k=1) == pytest.approx(0.75)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pass_at_k([], k=1)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            pass_at_k([[True]], k=0)


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


class TestEntrapmentReduction:
    def test_noise_helps(self):
        # noise off: 3/4 trapped = 0.75; noise on: 1/4 trapped = 0.25 → 0.5
        noise_off = [1.0, 1.0, 1.0, 0.0]
        noise_on = [0.0, 0.0, 0.0, 1.0]
        assert entrapment_reduction(noise_off, noise_on) == pytest.approx(0.5)

    def test_noise_no_effect(self):
        same = [1.0, 0.0, 1.0]
        assert entrapment_reduction(same, same) == pytest.approx(0.0)

    def test_noise_hurts(self):
        # noise on traps more → negative reduction
        noise_off = [0.0, 0.0]
        noise_on = [1.0, 1.0]
        assert entrapment_reduction(noise_off, noise_on) == pytest.approx(-1.0)

    def test_respects_custom_tol(self):
        noise_off = [0.05, 0.05]
        noise_on = [0.0, 0.0]
        # tol=0.09 → noise_off none trapped → reduction 0.0
        assert entrapment_reduction(noise_off, noise_on, tol=0.09) == pytest.approx(0.0)


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


class TestWilsonInterval:
    def test_zero_successes(self):
        lo, hi = wilson_interval(0, 10)
        assert lo == pytest.approx(0.0)
        assert 0.0 < hi < 0.5  # hand value: ~0.278

    def test_all_successes(self):
        lo, hi = wilson_interval(10, 10)
        assert hi == pytest.approx(1.0)
        assert 0.5 < lo < 1.0  # hand value: ~0.722

    def test_n_one(self):
        lo, hi = wilson_interval(0, 1)
        assert lo == pytest.approx(0.0)
        assert hi < 1.0
        lo, hi = wilson_interval(1, 1)
        assert hi == pytest.approx(1.0)
        assert lo > 0.0

    def test_brackets_point_estimate(self):
        lo, hi = wilson_interval(8, 10)
        assert lo < 0.8 < hi
        # hand-computed Wilson bounds for 8/10, z=1.96
        assert lo == pytest.approx(0.4901, abs=1e-3)
        assert hi == pytest.approx(0.9433, abs=1e-3)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            wilson_interval(0, 0)


class TestTwoProportionZ:
    def test_hand_computed_8_10_vs_2_10(self):
        # p1=0.8, p2=0.2, pooled=0.5, se=sqrt(0.5*0.5*(1/10+1/10))=sqrt(0.05)
        # z = 0.6/0.22360 = 2.6833, one-sided p ~ 0.00364
        z, p = two_proportion_z(8, 10, 2, 10)
        assert z == pytest.approx(2.6833, abs=1e-3)
        assert p == pytest.approx(0.00364, abs=1e-4)
        assert p < 0.05

    def test_equal_proportions(self):
        z, p = two_proportion_z(5, 10, 5, 10)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(0.5)

    def test_direction_antisymmetric(self):
        z_fwd, p_fwd = two_proportion_z(8, 10, 2, 10)
        z_rev, p_rev = two_proportion_z(2, 10, 8, 10)
        assert z_rev == pytest.approx(-z_fwd)
        assert z_fwd > 0 and z_rev < 0
        assert p_fwd + p_rev == pytest.approx(1.0)  # one-sided tails are complements

    def test_degenerate_pool_returns_neutral(self):
        # all failures (and all successes) give se=0 -> (0, 0.5) by convention
        assert two_proportion_z(0, 5, 0, 5) == (0.0, 0.5)
        assert two_proportion_z(5, 5, 5, 5) == (0.0, 0.5)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            two_proportion_z(1, 0, 1, 5)
        with pytest.raises(ValueError):
            two_proportion_z(1, 5, 1, 0)


class TestRestartBudgetCurve:
    def test_hand_built_indices(self):
        # first successes at restart 0, never, restart 2
        curve = restart_budget_curve([0, None, 2], k_max=3)
        assert [row["k"] for row in curve] == [1, 2, 3]
        assert [row["rate"] for row in curve] == [
            pytest.approx(1 / 3), pytest.approx(1 / 3), pytest.approx(2 / 3)
        ]
        assert all(row["n"] == 3 for row in curve)

    def test_ci_matches_wilson(self):
        curve = restart_budget_curve([0, None, 2], k_max=3)
        lo, hi = wilson_interval(1, 3)
        assert curve[0]["ci95"] == [pytest.approx(lo), pytest.approx(hi)]
        lo, hi = wilson_interval(2, 3)
        assert curve[2]["ci95"] == [pytest.approx(lo), pytest.approx(hi)]

    def test_monotone_nondecreasing(self):
        curve = restart_budget_curve([0, 1, 3, None, 2], k_max=5)
        rates = [row["rate"] for row in curve]
        assert rates == sorted(rates)
        assert rates[-1] == pytest.approx(0.8)

    def test_none_solved(self):
        curve = restart_budget_curve([None, None], k_max=2)
        assert all(row["rate"] == 0.0 for row in curve)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            restart_budget_curve([], k_max=1)

    def test_invalid_k_max_raises(self):
        with pytest.raises(ValueError):
            restart_budget_curve([0], k_max=0)


class TestDerivationVerifiability:
    def test_all_verify(self):
        solutions = [1, 2, 3]
        assert derivation_verifiability(solutions, lambda s: True) == pytest.approx(1.0)

    def test_none_verify(self):
        solutions = [1, 2, 3]
        assert derivation_verifiability(solutions, lambda s: False) == pytest.approx(0.0)

    def test_half_verify(self):
        solutions = [1, 2, 3, 4]
        assert derivation_verifiability(solutions, lambda s: s % 2 == 0) == pytest.approx(0.5)

    def test_empty(self):
        assert derivation_verifiability([], lambda s: True) == pytest.approx(0.0)
