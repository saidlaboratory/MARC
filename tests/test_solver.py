"""Tests for marc/eval/solver.py — the Solver contract, adapters and loader."""

import pytest

from marc.eval.problems import in_distribution
from marc.eval.solver import (
    FunctionSolver,
    GradientRefinementSolver,
    Solver,
    load_solver,
)


def test_gradient_solver_conforms_and_solves():
    solver = GradientRefinementSolver(seed=0)
    assert isinstance(solver, Solver)
    p = in_distribution(1)[0]
    cands = solver.sample(p, 3)
    assert len(cands) == 3
    # at least one restart should land on the exact solution
    assert any(
        all(abs(c - s) < 1e-6 for c, s in zip(cand, p.solution)) for cand in cands
    )


def test_function_solver_wraps_callable():
    p = in_distribution(1)[0]
    solver = FunctionSolver(lambda problem: problem.solution)
    assert isinstance(solver, Solver)
    assert solver.sample(p, 2) == [list(p.solution), list(p.solution)]


def test_function_solver_pass_graph():
    p = in_distribution(1)[0]
    solver = FunctionSolver(lambda g: [0.0] * len(g.variables), pass_graph=True)
    assert solver.sample(p, 1) == [[0.0, 0.0]]


def test_load_solver_names():
    assert isinstance(load_solver("refine"), GradientRefinementSolver)
    assert load_solver("dummy") is not None
    with pytest.raises(ValueError):
        load_solver("nope")


def test_load_solver_env(monkeypatch):
    monkeypatch.setenv("MARC_SOLVER", "refine")
    assert isinstance(load_solver(), GradientRefinementSolver)
