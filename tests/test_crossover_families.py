"""Smoke + property tests for the crossover-replication experiment.

Fast: tiny suite, tiny training. Verifies each separable family is well-formed (the
gold root is checker-accepted; the polish reaches it from a near-root start, so the
family is a fair proposal test not a polish wall), and that a --quick-scale run
produces the house-rules schema (random/lm/learned cells with CIs, both p-values)."""
import importlib.util
import random
import sys
from pathlib import Path

from marc.cas.checker import Checker
from marc.refine.iterative import refine

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "run_crossover_families", _root / "scripts" / "run_crossover_families.py")
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)
import run_dimension_scaling as rds  # noqa: E402


def test_families_gold_accepted_and_reachable():
    chk = Checker()
    for name, maker in rcf.FAMILIES.items():
        near = 0
        for j in range(10):
            g, sol, _ = maker(2, random.Random(j))
            assert chk.verify(g, [round(v, rds.DECIMALS) for v in sol]).accepted, name
            x0 = [s + random.Random(50 + j).uniform(-0.1, 0.1) for s in sol]
            near += rds.accepted(chk, g, refine(g, x0, noise=False, seed=0).x)
        # excluded quartic_well aside, each retained family has a reachable basin
        assert near >= 3, f"{name}: polish rarely reaches the root ({near}/10)"


def test_run_family_schema():
    maker = rcf.FAMILIES["baseline"]
    rows = rcf.run_family(maker, [1], K=2, ntest=6, epochs=1, ntrain=6, seeds=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 1
    for arm in ("random_restart", "lm", "learned_x0"):
        c = row[arm]
        assert c["n"] == 6 and 0 <= c["k"] <= 6
        lo, hi = c["ci95"]
        assert 0.0 <= lo <= c["rate"] <= hi <= 1.0
    assert 0.0 <= row["p_learned_gt_random"] <= 1.0
    assert 0.0 <= row["p_learned_gt_lm"] <= 1.0
