"""Smoke tests for scripts/run_coupled_eval.py — selftrain schema + checkpoint wiring.

Tiny sizes only; the real numbers come from the full script run. What matters here:
(a) the default (no ckpt) path keeps the house schema and self-trains as before,
(b) --ckpt routes the learned arm through load_solver("learned", ...) and never
    touches train_x0.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_coupled_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "run_coupled_eval.py",
)
rce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rce)


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
    for m in ("lm", "learned"):
        assert row[m]["n"] == 4 and 0 <= row[m]["k"] <= 4
    for p_key in ("p_learned_gt_random", "p_learned_gt_lm"):
        assert 0.0 <= row[p_key] <= 1.0


class _StubSolver:
    def __init__(self):
        self.calls = 0

    def sample(self, problem, k):
        self.calls += 1
        assert hasattr(problem, "graph") and len(problem.solution) == 2
        return [[0.0] * len(problem.solution) for _ in range(k)]


def test_ckpt_mode_routes_through_learned_solver(monkeypatch):
    stub = _StubSolver()
    seen = {}

    def fake_load(name, **kw):
        seen["name"] = name
        seen.update(kw)
        return stub

    monkeypatch.setattr(rce, "load_solver", fake_load)
    monkeypatch.setattr(rce, "train_x0", lambda *a, **kw: pytest.fail("train_x0 called in ckpt mode"))

    payload = rce.run(ns=[2], K=2, ntest=3, epochs=1, ntrain=4, ckpt="/fake/stage_a.pt")
    assert payload["learned_mode"] == "ckpt:stage_a.pt"
    assert seen["name"] == "learned"
    assert seen["checkpoint"] == "/fake/stage_a.pt"
    assert seen["polish"] is False  # hybrid_count's refine+Checker gate is the one polish step
    assert stub.calls > 0
