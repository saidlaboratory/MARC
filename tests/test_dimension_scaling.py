"""Smoke + property tests for the H1 dimension-scaling experiment (unified-v2).

Fast: tiny suite, tiny training. Verifies the trap family is constructed correctly
(the true root is checker-accepted; deterministic refine is trapped at the spurious
basin), that shapes scale with n, and that a --quick-equivalent run produces the
house-rules schema (k/n/rate/ci95 per method row, p-values, methodology tag). The
full learned-vs-classical result is produced by ``scripts/run_dimension_scaling.py``
(too heavy for the unit suite)."""
import importlib.util
import random
from pathlib import Path

from marc.cas.checker import Checker
from marc.refine.iterative import refine

_spec = importlib.util.spec_from_file_location(
    "run_dimension_scaling",
    Path(__file__).resolve().parent.parent / "scripts" / "run_dimension_scaling.py",
)
rds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rds)


def test_true_root_is_accepted():
    chk = Checker()
    rng = random.Random(0)
    for _ in range(5):
        g, sol, _ = rds.make_problem(3, rng)
        assert chk.verify(g, sol).accepted


def test_deterministic_refine_is_trapped():
    # every start sits in the spurious basin -> the shared noise-free refine never
    # reaches the root, and the checker gate rejects the trapped iterate
    chk = Checker()
    for g, sol, init in rds.suite(2, 8, seed=90000):
        assert not rds.accepted(chk, g, refine(g, init, noise=False, seed=0).x), \
            "deterministic refine should be fully trapped on this suite"


def test_refine_from_root_neighborhood_is_accepted():
    # the shared polisher + grid-snap + checker gate accepts a near-root start:
    # the acceptance criterion is achievable (not vacuously zero)
    chk = Checker()
    for j in range(5):
        g, sol, _ = rds.make_problem(2, random.Random(j))
        x0 = [s + random.Random(100 + j).uniform(-0.3, 0.3) for s in sol]
        assert rds.accepted(chk, g, refine(g, x0, noise=False, seed=0).x)


def test_problem_shape_scales_with_n():
    for n in (1, 4, 8):
        g, sol, init = rds.make_problem(n, random.Random(1))
        assert len(g.variables) == n
        assert len(g.factors) == n
        assert len(sol) == n and len(init) == n


def test_quick_run_schema():
    payload = rds.run(ns=[1], K=2, ntest=5, epochs=1, ntrain=4, seeds=2)
    assert payload["methodology"] == "unified-v2"
    assert "note" in payload and "seeds" in payload
    assert payload["K"] == 2 and payload["test_per_n"] == 5
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["n"] == 1
    for m in rds.METHODS:
        cell = row[m]
        assert cell["n"] == 10  # pooled over 2 seed replicates
        assert 0 <= cell["k"] <= cell["n"]
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
        assert len(cell["seed_rates"]) == 2
        assert 0.0 <= cell["seed_mean"] <= 1.0
        assert cell["seed_std"] >= 0.0
    for p_key in ("p_learned_gt_random", "p_learned_gt_langevin"):
        assert 0.0 <= row[p_key] <= 1.0
