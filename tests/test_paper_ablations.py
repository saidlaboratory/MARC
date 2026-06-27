"""The checkpoint-gated ablations must skip cleanly (not crash) without weights."""

import os

import pytest

from marc.eval.ablations import guidance_ablation, purist_ablation


@pytest.fixture(autouse=True)
def _no_checkpoints(monkeypatch):
    monkeypatch.delenv("MARC_CKPT", raising=False)
    monkeypatch.delenv("MARC_CKPT_PURIST", raising=False)


def test_guidance_skips_without_checkpoint():
    out = guidance_ablation.run_ablation(weights=[0.0, 1.0], k=2, n_ho=4)
    assert out["ablation"] == "guidance"
    assert out["status"] == "skipped"
    assert "checkpoint" in out["reason"].lower()


def test_purist_skips_without_checkpoint():
    out = purist_ablation.run_ablation(n=4, k=2)
    assert out["ablation"] == "purist_reward"
    assert out["status"] == "skipped"


def test_purist_skips_when_only_standard_present(tmp_path, monkeypatch):
    # standard present but purist missing -> still a clean skip, not a crash
    ckpt = tmp_path / "std.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setenv("MARC_CKPT", str(ckpt))
    out = purist_ablation.run_ablation(n=4, k=2)
    assert out["status"] == "skipped"
    assert "purist" in out["reason"].lower()
