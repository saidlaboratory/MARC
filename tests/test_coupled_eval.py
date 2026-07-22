"""Smoke tests for scripts/run_coupled_eval.py — selftrain schema + checkpoint wiring.

Tiny sizes only; the real numbers come from the full script run. What matters here:
(a) the default (no ckpt) path keeps the house schema and self-trains as before,
(b) --ckpt routes the learned arm through load_solver("learned", ...) and never
    touches train_x0.
"""
import pytest

from conftest import StubSolver, load_script, patch_load_solver

rce = load_script("run_coupled_eval")


def test_quick_schema_selftrain():
    payload = rce.run(ns=[2], K=2, ntest=4, epochs=1, ntrain=4)
    assert payload["learned_mode"] == "selftrain"
    assert payload["K"] == 2 and payload["test_per_n"] == 4 and payload["epochs"] == 1
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["n"] == 2
    for m in ("langevin", "random", "lm", "learned"):
        cell = row[m]
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
        assert cell["wall_ms_total"] >= 0.0
        assert cell["wall_ms_mean"] >= 0.0
    for m in ("lm", "learned"):
        assert row[m]["n"] == 4 and 0 <= row[m]["k"] <= 4
    for p_key in ("p_learned_gt_random", "p_learned_gt_lm"):
        assert 0.0 <= row[p_key] <= 1.0


def _zeros(problem, k, call):
    assert hasattr(problem, "graph") and len(problem.solution) == 2
    return [[0.0] * len(problem.solution) for _ in range(k)]


def test_ckpt_mode_routes_through_learned_solver(monkeypatch):
    stub = StubSolver(_zeros)
    seen = patch_load_solver(monkeypatch, rce, stub)
    monkeypatch.setattr(rce, "train_x0", lambda *a, **kw: pytest.fail("train_x0 called in ckpt mode"))

    payload = rce.run(ns=[2], K=2, ntest=3, epochs=1, ntrain=4, ckpt="/fake/stage_a.pt")
    assert payload["learned_mode"] == "ckpt:stage_a.pt"
    assert seen["name"] == "learned"
    assert seen["checkpoint"] == "/fake/stage_a.pt"
    assert seen["polish"] is False  # hybrid_count's refine+Checker gate is the one polish step
    assert stub.calls > 0
