"""Smoke tests for scripts/run_hard_eval.py: default self-train mode keeps the
output schema, and --ckpt routes the learned arm through LearnedSolver instead
of train_x0. Heavy training is stubbed; the full result comes from the script."""
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_hard_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "run_hard_eval.py",
)
rhe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rhe)


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_hard_eval.py"] + argv)
    monkeypatch.setattr(rhe, "HARD_TEMPLATES_EXT", rhe.HARD_TEMPLATES_EXT[:1])
    rhe.main()
    return json.loads((tmp_path / "results/p_hard/hard_eval.json").read_text())


def test_selftrain_schema(monkeypatch, tmp_path):
    monkeypatch.delenv("MARC_CKPT", raising=False)
    real = rhe.train_x0
    monkeypatch.setattr(rhe, "train_x0", lambda items, epochs: real(items[:2], 1))
    payload = _run(monkeypatch, tmp_path, ["--quick", "--K", "1", "--test", "2"])
    assert payload["learned_mode"] == "selftrain"
    assert payload["K"] == 1 and payload["test_per_family"] == 2
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    for m in ("refine_cold", "refine_langevin", "random_restart", "lm", "learned_hybrid"):
        cell = row[m]
        assert cell["n"] == 2 and 0 <= cell["k"] <= 2
        lo, hi = cell["ci95"]
        assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0
    assert isinstance(row["hybrid_beats_langevin_sig"], bool)
    assert 0.0 <= row["p_learned_gt_lm"] <= 1.0


def test_ckpt_mode_uses_learned_solver(monkeypatch, tmp_path):
    calls = {}

    class StubSolver:
        def sample(self, problem, k):
            calls["sampled"] = calls.get("sampled", 0) + k
            return [list(problem.solution)] * k

    def fake_load_solver(name, **kwargs):
        calls["name"] = name
        calls["kwargs"] = kwargs
        return StubSolver()

    monkeypatch.setattr(rhe, "load_solver", fake_load_solver)
    monkeypatch.setattr(rhe, "train_x0",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("train_x0 called in ckpt mode")))
    payload = _run(monkeypatch, tmp_path,
                   ["--quick", "--K", "1", "--test", "2", "--ckpt", "fake_stage_a.pt"])
    assert calls["name"] == "learned"
    assert calls["kwargs"] == {"checkpoint": "fake_stage_a.pt", "polish": False}
    assert calls["sampled"] >= 1
    assert payload["learned_mode"] == "ckpt:fake_stage_a.pt"
    # stub proposes the true solution -> the shared polish/accept path solves everything
    assert payload["rows"][0]["learned_hybrid"]["k"] == 2
