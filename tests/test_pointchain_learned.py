"""Smoke + property tests for the point-chain LEARNED eval (law's geometry prediction).

Fast: tiny suite, tiny training. Verifies the point-chain family is well-formed for the
denoiser (gold solution is checker-accepted after the grid snap; deterministic descent
from a random start is not vacuously perfect), that training runs and the schema matches
house rules (k/n/rate/ci95 per arm, a learned>random p-value, per-seed sd with seeds>1).
The full learned-vs-random result is produced by scripts/run_pointchain_learned.py.
"""
import importlib.util
import random
from pathlib import Path

from marc.cas.checker import Checker

_spec = importlib.util.spec_from_file_location(
    "run_pointchain_learned",
    Path(__file__).resolve().parent.parent / "scripts" / "run_pointchain_learned.py",
)
rpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpl)


def test_gold_solution_accepted():
    chk = Checker()
    for g, sol in rpl.suite(3, 5, seed0=0):
        assert rpl.accepted(chk, g, sol)


def test_suite_shapes_scale_with_k():
    for k in (1, 2, 4):
        items = rpl.suite(k, 4, seed0=1)
        assert len(items) == 4
        for g, sol in items:
            assert len(g.variables) == 2 * k
            assert len(sol) == 2 * k


def test_quick_run_schema():
    payload = rpl.run([1], trials=6, K=2, epochs=1, ntrain=6, seeds=2)
    assert payload["seeds"] == 2 and payload["K"] == 2
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["points"] == 1 and row["n"] == 2
    for m in ("langevin", "random_restart", "learned"):
        cell = row[m]
        assert cell["n"] == 12  # pooled over 2 seeds x 6 trials
        assert 0 <= cell["k"] <= cell["n"]
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
        assert len(cell["seed_rates"]) == 2 and cell["seed_sd"] >= 0.0
    assert 0.0 <= row["p_learned_gt_random"] <= 1.0
