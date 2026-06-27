"""Tests for run_split_eval — real structural-split generalization gap."""

import json

import pytest

from marc.eval.problems import held_out_structure, in_distribution
from marc.eval.runner import run_split_eval
from marc.eval.solver import GradientRefinementSolver


def test_split_eval_reports_real_gap_and_is_serialisable():
    solver = GradientRefinementSolver(seed=0)
    metrics = run_split_eval(
        in_distribution(8),
        held_out_structure(8),
        solver=solver,
        n_samples=2,
        solver_name="refine",
    )
    # gradient solver derives, so it solves both splits -> zero gap
    assert metrics["splits"]["in_distribution"]["solve_rate"] == 1.0
    assert metrics["splits"]["held_out_structure"]["solve_rate"] == 1.0
    assert metrics["generalization_gap"] == 0.0
    assert metrics["solver"] == "refine"
    assert len(metrics["per_problem"]) == 16
    json.dumps(metrics)  # must not raise


def test_split_eval_gap_tracks_solver_weakness():
    # a solver that nails the 2-var in-dist structure but is wrong on the 3-var
    # held-out structure -> positive generalization gap.
    class StructureBiasedSolver:
        def sample(self, problem, k):
            if len(problem.solution) == 2:
                return [list(problem.solution)] * k
            wrong = [v + 1.0 for v in problem.solution]  # right length, wrong values
            return [wrong] * k

    metrics = run_split_eval(
        in_distribution(6),
        held_out_structure(6),
        solver=StructureBiasedSolver(),
        n_samples=1,
    )
    assert metrics["splits"]["in_distribution"]["solve_rate"] == 1.0
    assert metrics["splits"]["held_out_structure"]["solve_rate"] == 0.0
    assert metrics["generalization_gap"] == 1.0


def test_split_eval_requires_nonempty_splits():
    with pytest.raises(ValueError):
        run_split_eval([], held_out_structure(2))
