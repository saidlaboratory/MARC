"""Smoke tests for the cross-family (LOFO) eval script: the default self-train
schema is unchanged, and --ckpt mode routes through LearnedSolver and flags the
contamination caveat (a checkpoint trained on all families breaks leave-one-out)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_crossfamily_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "run_crossfamily_eval.py",
)
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)

FAMS = rcf.HARD_TEMPLATES_EXT[:2]


def test_selftrain_schema_unchanged():
    payload = rcf.run(FAMS, K=1, ntest=2, epochs=1, ntrain=2)
    assert payload["learned_mode"] == "selftrain"
    assert "ckpt_mode_caveat" not in payload
    assert payload["K"] == 1 and payload["test_per_family"] == 2
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert isinstance(row["trained_on"], list) and row["held_out"] not in row["trained_on"]
        for m in ("refine_langevin", "learned_cross"):
            cell = row[m]
            assert 0 <= cell["k"] <= cell["n"] == 2
            lo, hi = cell["ci95"]
            assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
        assert 0.0 <= row["p_hybrid_gt_langevin"] <= 1.0


class _StubSolver:
    calls = 0

    def __init__(self, ckpt):
        self.ckpt = ckpt

    def sample(self, problem, k):
        _StubSolver.calls += 1
        return [[0.0] * len(problem.graph.variables)] * k


def test_ckpt_mode_routes_through_solver_and_flags_caveat(monkeypatch):
    monkeypatch.setattr(rcf, "LearnedSolver", _StubSolver)
    _StubSolver.calls = 0
    payload = rcf.run(FAMS, K=1, ntest=2, epochs=1, ntrain=2,
                      ckpt="/tmp/denoiser_stage_a.pt")
    assert payload["learned_mode"] == "ckpt:denoiser_stage_a.pt"
    assert "contamination" in payload["ckpt_mode_caveat"]
    assert _StubSolver.calls == 4  # 2 families x 2 test instances
    for row in payload["rows"]:
        assert row["trained_on"] == "all_families_ckpt"
