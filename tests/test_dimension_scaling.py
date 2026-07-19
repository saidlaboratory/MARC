"""Smoke + property tests for the H1 dimension-scaling experiment.

Fast: tiny suite, no training. Verifies the trap family is constructed correctly
(the true root is checker-accepted; deterministic line-search descent is trapped at
the spurious basin) and that shapes scale with n. The full learned-vs-classical
result is produced by ``scripts/run_dimension_scaling.py`` (too heavy for the unit
suite)."""
import importlib.util
import random
from pathlib import Path

from marc.cas.checker import Checker

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


def test_deterministic_descent_is_trapped():
    # every start sits in the spurious basin -> plain descent never reaches the root
    solved = 0
    for g, sol, init in rds.suite(2, 8, seed=90000):
        cas = rds.cas_for(g)
        if rds.close(rds.descend(cas, init, steps=400), sol):
            solved += 1
    assert solved == 0, "deterministic descent should be fully trapped on this suite"


def test_problem_shape_scales_with_n():
    for n in (1, 4, 8):
        g, sol, init = rds.make_problem(n, random.Random(1))
        assert len(g.variables) == n
        assert len(g.factors) == n
        assert len(sol) == n and len(init) == n
