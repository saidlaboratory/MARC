"""Tests for the classical solver baselines (W5): exact linear + scipy LM.

Covers: build_residual_jac vs finite differences, ScipySolver on a hard
(nonlinear) family, ExactLinearSolver exact-on-linear / empty-on-nonlinear,
load_solver registration, and the new eval-row schema fragment.
"""

import numpy as np

from marc.cas.checker import Checker
from marc.data.templates import BilinearSystemTemplate, LinearSystem2x2Template
from marc.eval.metrics import wilson_interval
from marc.eval.runner import Problem
from marc.eval.solver import ExactLinearSolver, ScipySolver, Solver, load_solver
from marc.refine.iterative import build_residual_jac

from conftest import load_script


def _problem(template, seed=0):
    g, sol = template.generate(seed=seed)
    return Problem(id=template.name, graph=g, solution=[float(v) for v in sol.values()])


def test_jacobian_matches_finite_differences():
    g, _ = BilinearSystemTemplate().generate(seed=0)
    r_fn, j_fn, n = build_residual_jac(g)
    x = 0.7 + 0.3 * np.arange(n)  # generic non-symmetric point
    r0 = np.asarray(r_fn(*x), dtype=float).reshape(-1)
    J = np.asarray(j_fn(*x), dtype=float).reshape(len(r0), n)
    eps = 1e-6
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fd = (
            np.asarray(r_fn(*xp), dtype=float) - np.asarray(r_fn(*xm), dtype=float)
        ).reshape(-1) / (2 * eps)
        assert np.allclose(J[:, j], fd, atol=1e-5)


def test_scipy_solver_solves_hard_family():
    p = _problem(BilinearSystemTemplate(), seed=3)
    solver = ScipySolver(seed=0)
    assert isinstance(solver, Solver)
    cands, infos = solver.sample_with_info(p, 8)
    assert len(cands) == 8 and len(infos) == 8
    chk = Checker()
    assert any(chk.verify(p.graph, c).accepted for c in cands)
    for info in infos:
        assert {"n_steps", "final_energy", "converged"} <= set(info)
        assert info["n_steps"] >= 1
        assert info["converged"] == (info["final_energy"] <= 1e-6)


def test_scipy_sample_and_sample_with_info_share_one_code_path():
    p = _problem(BilinearSystemTemplate(), seed=1)
    plain = ScipySolver(seed=0).sample(p, 3)
    with_info, _ = ScipySolver(seed=0).sample_with_info(p, 3)
    assert plain == with_info  # identical candidates under the same seed


def test_exact_solver_exact_on_linear_system():
    p = _problem(LinearSystem2x2Template(), seed=0)
    solver = ExactLinearSolver()
    assert isinstance(solver, Solver)
    cands, infos = solver.sample_with_info(p, 3)
    assert len(cands) == 3  # deterministic: k copies of the one solution
    assert cands[0] == cands[1] == cands[2]
    chk = Checker()
    assert all(chk.verify(p.graph, c).accepted for c in cands)
    assert all(i["converged"] and i["final_energy"] <= 1e-12 for i in infos)


def test_exact_solver_rate_one_across_linear_instances():
    chk = Checker()
    solver = ExactLinearSolver()
    for seed in range(10):
        p = _problem(LinearSystem2x2Template(), seed=seed)
        cands = solver.sample(p, 1)
        assert cands and chk.verify(p.graph, cands[0]).accepted  # 1.0 solve rate


def test_exact_solver_no_candidates_on_nonlinear():
    p = _problem(BilinearSystemTemplate(), seed=0)
    assert ExactLinearSolver().sample(p, 4) == []
    assert ExactLinearSolver().sample_with_info(p, 4) == ([], [])


def test_load_solver_roundtrip():
    assert isinstance(load_solver("lm"), ScipySolver)
    assert isinstance(load_solver("exact"), ExactLinearSolver)


def test_hard_eval_lm_row_schema():
    mod = load_script("run_hard_eval")
    items = mod.gen(BilinearSystemTemplate(), 3, seed0=0)
    k, n = mod.lm_count(items, 4)
    assert n == 3 and 0 <= k <= n
    row = {"k": k, "n": n, "rate": k / n, "ci95": wilson_interval(k, n)}
    assert set(row) == {"k", "n", "rate", "ci95"}
    lo, hi = row["ci95"]
    assert 0.0 <= lo <= row["rate"] <= hi <= 1.0
