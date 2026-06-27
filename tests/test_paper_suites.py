"""Tests for the P2 paper suites — the JSON contract plot_results.py reads."""

from marc.eval.paper.suites import (
    run_generalization_gap,
    run_length_extrapolation,
    run_perturbation,
)
from marc.eval.solver import GradientRefinementSolver


def _solver():
    # small, fast, deterministic real solver
    return GradientRefinementSolver(seed=0)


def test_generalization_gap_contract():
    out = run_generalization_gap(_solver(), n_id=4, n_ho=4, k=2)
    assert out["suite"] == "generalization_gap"
    assert "generalization_gap" in out
    for split in ("in_distribution", "held_out_structure"):
        assert "solve_rate" in out["splits"][split]
        assert "perturbation_robustness" in out["splits"][split]


def test_perturbation_sweep_contract():
    deltas = [0.0, 0.5, 1.0]
    out = run_perturbation(_solver(), deltas=deltas, n_id=4, n_ho=4, k=2)
    assert out["suite"] == "perturbation"
    assert [r["delta"] for r in out["sweep"]] == deltas
    for row in out["sweep"]:
        for split in ("in_distribution", "held_out_structure"):
            cell = row[split]
            assert {"solve_rate", "robustness", "perturbed_solve_rate"} <= cell.keys()


def test_length_extrapolation_contract():
    lengths = [2, 3, 4]
    out = run_length_extrapolation(_solver(), lengths=lengths, n=4, k=2)
    assert out["suite"] == "length_extrapolation"
    assert [r["length"] for r in out["sweep"]] == lengths
    # the real gradient solver is an oracle on this clean range
    for row in out["sweep"]:
        assert row["solve_rate"] == 1.0, row
        assert row["extrapolation"] == (row["length"] > out["train_length"])
