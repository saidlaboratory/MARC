"""Smoke + schema tests for the point-chain learned-vs-random eval (R9 follow-up).

Fast: tiny trial counts, k=1 only. Verifies the house-rules schema (k/n/rate/ci95
per arm, z-test when the learned arm runs), the skipped-learned path without a
checkpoint, and that a checkpoint routes through load_solver with polish off. The
real result is produced by ``scripts/run_pointchain_eval.py`` (too heavy here)."""
from conftest import StubSolver, diverge_then_zeros, load_script, patch_load_solver

rpe = load_script("run_pointchain_eval")


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


def test_ckpt_mode_uses_stub_solver(monkeypatch):
    stub = StubSolver(diverge_then_zeros)
    seen = patch_load_solver(monkeypatch, rpe, stub)
    payload = rpe.run([1], trials=3, K=2, ckpt="stub.pt")
    assert seen["name"] == "learned"
    assert seen["checkpoint"] == "stub.pt"
    assert seen["polish"] is False  # the shared refine stays the one polisher
    assert stub.calls > 0
    assert payload["learned_mode"] == "checkpoint" and payload["ckpt"] == "stub.pt"
    row = payload["rows"][0]
    assert 0 <= row["learned"]["k"] <= row["learned"]["n"] == 3
    assert 0.0 <= row["p_learned_gt_random"] <= 1.0
