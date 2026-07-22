"""Tests for marc/eval/solver.py — the Solver contract, adapters and loader."""

import pytest

from marc.eval.problems import in_distribution
from marc.eval.solver import (
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


def test_gradient_solver_sample_with_info():
    p = in_distribution(1)[0]
    cands, infos = GradientRefinementSolver(seed=0).sample_with_info(p, 3)
    assert len(cands) == 3 and len(infos) == 3
    for info in infos:
        assert {"n_steps", "best_energy", "final_energy", "converged", "energies"} <= set(info)
        assert 1 <= len(info["energies"]) <= 50
        assert info["best_energy"] <= info["energies"][0]  # refinement went downhill
        assert info["best_energy"] <= min(info["energies"])  # best over the whole trace
        assert info["n_steps"] >= len(info["energies"]) - 1  # downsampled, not truncated


def test_sample_and_sample_with_info_share_one_code_path():
    p = in_distribution(1)[0]
    plain = GradientRefinementSolver(seed=0).sample(p, 2)
    with_info, _ = GradientRefinementSolver(seed=0).sample_with_info(p, 2)
    assert plain == with_info  # identical candidates under the same seed


def test_load_solver_names():
    assert isinstance(load_solver("refine"), GradientRefinementSolver)
    assert load_solver("dummy") is not None
    with pytest.raises(ValueError):
        load_solver("nope")


def test_load_solver_env(monkeypatch):
    monkeypatch.setenv("MARC_SOLVER", "refine")
    assert isinstance(load_solver(), GradientRefinementSolver)
