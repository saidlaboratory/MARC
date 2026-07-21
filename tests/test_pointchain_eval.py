"""Smoke + schema tests for the point-chain learned-vs-random eval (R9 follow-up).

Fast: tiny trial counts, k=1 only. Verifies the house-rules schema (k/n/rate/ci95
per arm, z-test when the learned arm runs), the skipped-learned path without a
checkpoint, and that a checkpoint routes through load_solver with polish off. The
real result is produced by ``scripts/run_pointchain_eval.py`` (too heavy here)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_pointchain_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "run_pointchain_eval.py",
)
rpe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpe)


def test_skipped_learned_schema():
    payload = rpe.run([1], trials=3, K=2, ckpt=None)
    assert payload["learned_mode"] == "skipped" and payload["ckpt"] is None
    assert payload["K"] == 2 and payload["trials"] == 3
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["points"] == 1 and row["n"] == 2
    for arm in ("langevin", "random_restart"):
        cell = row[arm]
        assert cell["n"] == 3 and 0 <= cell["k"] <= 3
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
    assert row["learned"]["status"] == "skipped"
    assert "p_learned_gt_random" not in row


class _StubSolver:
    def __init__(self):
        self.calls = 0

    def sample(self, problem, k):
        self.calls += 1
        nv = len(problem.graph.variables)
        # first call diverges (None candidate), the rest propose zeros
        return [None if self.calls == 1 else [0.0] * nv for _ in range(k)]


def test_ckpt_mode_uses_stub_solver(monkeypatch):
    stub = _StubSolver()
    seen = {}

    def fake_load(name, **kw):
        seen.update(kw, name=name)
        return stub

    monkeypatch.setattr(rpe, "load_solver", fake_load)
    payload = rpe.run([1], trials=3, K=2, ckpt="stub.pt")
    assert seen["name"] == "learned"
    assert seen["checkpoint"] == "stub.pt"
    assert seen["polish"] is False  # the shared refine stays the one polisher
    assert stub.calls > 0
    assert payload["learned_mode"] == "checkpoint" and payload["ckpt"] == "stub.pt"
    row = payload["rows"][0]
    assert 0 <= row["learned"]["k"] <= row["learned"]["n"] == 3
    assert 0.0 <= row["p_learned_gt_random"] <= 1.0
