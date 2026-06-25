"""Tests for the noise on/off entrapment ablation (RQ2)."""

import json

from marc.eval.ablations.noise_ablation import run_ablation, write_outputs


def test_ablation_shows_noise_reduces_entrapment():
    summary = run_ablation(n_graphs=12, seeds=[0, 1, 2])
    # deterministic descent is trapped by construction on every graph
    assert summary["entrapment_rate_noise_off"] == 1.0
    # noise must strictly help on this suite
    assert summary["entrapment_rate_noise_on_mean"] < 1.0
    assert summary["entrapment_reduction_mean"] > 0.0
    assert summary["noise_helps"] is True


def test_ablation_writes_report_and_summary(tmp_path):
    summary = run_ablation(n_graphs=8, seeds=[0, 1])
    report = write_outputs(summary, tmp_path)
    assert report.exists()
    text = report.read_text()
    assert "Entrapment" in text or "entrapment" in text
    # summary.json is written and machine-readable, without the un-serialisable arms
    data = json.loads((tmp_path / "summary.json").read_text())
    assert "entrapment_reduction_mean" in data
    assert not any(k.startswith("_") for k in data)
