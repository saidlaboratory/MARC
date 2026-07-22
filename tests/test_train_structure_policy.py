"""Tests for scripts/train_structure_policy.py (W7: seed contract, reward-weighted
CE, unsolvable filtering, ablate/reference-solver guards)."""

import importlib.util
import os
import subprocess
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "train_structure_policy.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("train_structure_policy", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ckpt_records_seed_space(tmp_path):
    out = tmp_path / "sp.pt"
    r = subprocess.run(
        [sys.executable, SCRIPT, "--out", str(out), "--epochs", "1",
         "--n-train", "8", "--T", "6", "--D", "16"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    tc = torch.load(out, map_location="cpu", weights_only=False)["train_config"]
    assert tc["seed_space_version"] == 1
    assert tc["train_seed_range"] == [0, 8]
    assert tc["val_seed_range"] == [500000, 500050]
    assert tc["test_seed_min"] == 900000
    from marc.structure import invention_data
    assert tc["data_version"] == getattr(invention_data, "DATA_VERSION", 1)


def test_seed_guard_val_crossing_test_min(mod, tmp_path):
    # 400000 + 500000 + 50 > 900000
    with pytest.raises(SystemExit, match="seed-space"):
        mod.main(["--out", str(tmp_path / "x.pt"), "--seed", "400000"])
    with pytest.raises(SystemExit, match="seed-space"):
        mod.main(["--out", str(tmp_path / "x.pt"), "--n-train", "600000"])


def test_rl_gradient_direction(mod):
    scores = torch.zeros(3, 5, requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 0.0])
    loss = mod.rl_loss_from_scores(list(scores), [0, 1, 2], rewards, weight=1.0)
    loss.backward()
    g = scores.grad
    # the descent update (-grad) raises the picked logit of the above-mean pick...
    assert g[0, 0] < 0
    # ...and lowers the picked logits of below-mean picks
    assert g[1, 1] > 0 and g[2, 2] > 0


def test_rl_all_equal_rewards_zero_loss(mod):
    scores = torch.randn(4, 5, requires_grad=True)
    loss = mod.rl_loss_from_scores(list(scores), [0, 1, 2, 3], torch.ones(4), weight=0.7)
    assert float(loss.detach()) == 0.0


def test_filter_unsolvable_drops_and_records(mod, tmp_path, monkeypatch):
    # craft "gold does not solve" for exactly the instance with seed 0
    monkeypatch.setattr(mod, "candidate_solves",
                        lambda inst, pick, solver, checker, cache: inst.seed != 0)
    out = tmp_path / "f.pt"
    mod.main(["--out", str(out), "--epochs", "1", "--n-train", "8", "--T", "6",
              "--D", "16", "--filter-unsolvable"])
    tc = torch.load(out, map_location="cpu", weights_only=False)["train_config"]
    assert tc["filtered"] == {"checked": 8, "dropped": 1}


def _args(**kw):
    base = dict(D=16, L=1, K=4, ablate_context=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_ablate_guard_without_kwarg(mod, monkeypatch):
    class NoAblate:
        def __init__(self, D=64, L=2, K=4):
            pass

    monkeypatch.setattr(mod, "StructurePolicy", NoAblate)
    with pytest.raises(SystemExit, match="unit W3"):
        mod.build_model_kwargs(_args())


def test_ablate_kwarg_written_only_when_set(mod, monkeypatch):
    class WithAblate:
        def __init__(self, D=64, L=2, K=4, ablate_context=False):
            pass

    monkeypatch.setattr(mod, "StructurePolicy", WithAblate)
    assert mod.build_model_kwargs(_args())["ablate_context"] is True
    assert "ablate_context" not in mod.build_model_kwargs(_args(ablate_context=False))


def test_reference_solver_literal(mod):
    # owned by invention_data: training reward, eval arms, and certification
    # grade with the same object by construction
    from marc.structure.invention_data import REFERENCE_SOLVER

    assert mod.REFERENCE_SOLVER is REFERENCE_SOLVER
    assert mod.REFERENCE_SOLVER == {"name": "lm", "k_refine": 4}


def test_rl_path_runs(mod, tmp_path, monkeypatch):
    # reward wiring only — solve check stubbed to "gold solves"
    monkeypatch.setattr(mod, "candidate_solves",
                        lambda inst, pick, solver, checker, cache: pick == inst.gold_idx)
    out = tmp_path / "rl.pt"
    mod.main(["--out", str(out), "--epochs", "1", "--n-train", "4", "--T", "6",
              "--D", "16", "--rl-weight", "0.5"])
    assert out.exists()
