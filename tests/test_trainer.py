"""Tests for marc/train/trainer.py — the config-driven scale trainer."""
import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, HeteroData

from marc.data.collate import collate_fn
from marc.diffusion.schedule import cosine_beta_schedule
from marc.train.trainer import (
    CSV_HEADER,
    DEFAULTS,
    RunLogger,
    filter_kwargs,
    load_config,
    main,
    resolve_device,
    resolve_templates,
    stage_a_loss_local,
    train_stage_a_scaled,
)

REPO_YAML = Path(__file__).resolve().parents[1] / "marc" / "configs" / "train" / "scale.yaml"


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------

class StubDenoiser(nn.Module):
    """Minimal denoiser (no marc.model dependency, no supports_per_graph_t)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.calls = 0
        self.t_numels = []

    def forward(self, data, t):
        self.calls += 1
        self.t_numels.append(int(t.numel()))
        return self.linear(data["variable"].x)


class PerGraphStub(StubDenoiser):
    """Stub advertising the C1 per-graph-t contract; asserts t is a vector."""

    def __init__(self, expected_num_graphs):
        super().__init__()
        self.supports_per_graph_t = True
        self.expected_num_graphs = expected_num_graphs

    def forward(self, data, t):
        assert t.numel() == self.expected_num_graphs, (
            f"expected per-graph t of {self.expected_num_graphs}, got {t.numel()}"
        )
        return super().forward(data, t)


def make_item(n_vars=2):
    d = HeteroData()
    d["variable"].x = torch.zeros(n_vars, 1)
    d["factor"].x = torch.zeros(2, 1)
    d["variable", "connected_to", "factor"].edge_index = torch.tensor(
        [[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long
    )
    d["variable", "connected_to", "factor"].edge_attr = torch.ones(4, 1)
    return d, torch.randn(n_vars, 1)


def tiny_cfg(epochs_a=2):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["training"].update(
        {"T": 50, "epochs_A": epochs_a, "batch_size": 2, "num_workers": 0,
         "warmup_frac": 0.0, "device": "cpu"}
    )
    cfg["checkpointing"]["save_every_n_epochs"] = 1
    return cfg


def run_tiny_training(out_dir, epochs_a=2, model=None, resume_path=None):
    cfg = tiny_cfg(epochs_a)
    items = [make_item() for _ in range(4)]
    loader = DataLoader(items, batch_size=2, collate_fn=collate_fn)
    _, alpha_bar = cosine_beta_schedule(cfg["training"]["T"])
    model = model or StubDenoiser()
    logger = RunLogger(str(out_dir))
    train_stage_a_scaled(
        model, loader, alpha_bar, cfg, "cpu", str(out_dir), logger,
        model_kwargs={"D": 1, "L": 1, "step_dim": 1}, resume_path=resume_path,
    )
    return model


# ---------------------------------------------------------------------------
# Device / config / templates
# ---------------------------------------------------------------------------

def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto():
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_load_config_fills_defaults(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text("model:\n  D: 64\ntraining:\n  epochs_A: 3\n  device: \"cuda\"\n")
    cfg = load_config(str(p))
    assert cfg["model"]["D"] == 64
    assert cfg["model"]["L"] == DEFAULTS["model"]["L"]
    assert cfg["training"]["epochs_A"] == 3
    assert cfg["training"]["device"] == "cuda"
    # new keys, all defaulted in code
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["amp"] == "auto"
    assert cfg["training"]["grad_clip"] == 1.0
    assert cfg["training"]["warmup_frac"] == 0.03
    assert cfg["training"]["num_workers"] == 2
    assert cfg["data"]["templates"] == DEFAULTS["data"]["templates"]
    assert cfg["data"]["dir"] == "results/scale/train_data"
    assert cfg["grpo"]["problems_subset"] == 256


def test_load_config_cli_overrides(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text("training:\n  device: \"cuda\"\n")
    cfg = load_config(str(p), {"training": {"device": "cpu"}})
    assert cfg["training"]["device"] == "cpu"


def test_all_yaml_templates_resolve():
    templates = resolve_templates(DEFAULTS["data"]["templates"])
    assert len(templates) == 8
    assert {t.name for t in templates} == set(DEFAULTS["data"]["templates"])


def test_unknown_template_warns_and_skips():
    with pytest.warns(UserWarning, match="unknown template 'Bogus'"):
        templates = resolve_templates(["LinearSystem2x2", "Bogus"])
    assert [t.name for t in templates] == ["LinearSystem2x2"]


def test_all_unknown_templates_raises():
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            resolve_templates(["Bogus"])


# ---------------------------------------------------------------------------
# Stage-A loss seam (cross-unit contract C1)
# ---------------------------------------------------------------------------

def test_stage_a_loss_fallback_loop():
    items = [make_item() for _ in range(3)]
    data, solutions = collate_fn(items)
    _, alpha_bar = cosine_beta_schedule(100)
    model = StubDenoiser()  # no supports_per_graph_t flag
    loss = stage_a_loss_local(model, data, solutions, alpha_bar, 100, "cpu")
    assert torch.isfinite(loss)
    assert model.calls == 3  # per-graph loop, one forward per problem
    assert all(n == 1 for n in model.t_numels)  # scalar-ish t per graph


def test_stage_a_loss_batched_dispatch():
    items = [make_item() for _ in range(3)]
    data, solutions = collate_fn(items)
    _, alpha_bar = cosine_beta_schedule(100)
    model = PerGraphStub(expected_num_graphs=3)
    loss = stage_a_loss_local(model, data, solutions, alpha_bar, 100, "cpu")
    assert torch.isfinite(loss)
    assert model.calls == 1  # one batched forward
    assert model.t_numels == [3]  # per-graph timestep vector


def test_stage_a_loss_flagged_model_plain_heterodata_falls_back():
    # a single HeteroData (no .batch vector) must take the loop path even if
    # the model advertises per-graph t
    data, solution = make_item()
    _, alpha_bar = cosine_beta_schedule(100)
    model = PerGraphStub(expected_num_graphs=1)
    loss = stage_a_loss_local(model, data, [solution], alpha_bar, 100, "cpu")
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Checkpoints / resume / logs
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_resume(tmp_path):
    run_tiny_training(tmp_path, epochs_a=2)
    ckpt = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ckpt["epoch"] == 2
    assert ckpt["stage"] == "a"
    assert ckpt["global_step"] == 4  # 2 epochs x 2 batches
    assert ckpt["optimizer_state_dict"]["state"], "optimizer state should be non-empty"

    # resume from latest.pt with a larger epoch budget: counter continues
    run_tiny_training(tmp_path, epochs_a=4, resume_path=str(tmp_path / "latest.pt"))
    ckpt2 = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ckpt2["epoch"] == 4
    assert ckpt2["global_step"] == 8
    assert ckpt2["optimizer_state_dict"]["state"]


def test_checkpoint_learned_solver_compat(tmp_path):
    run_tiny_training(tmp_path, epochs_a=1)
    for name in ("latest.pt", "best.pt", "epoch_0001.pt"):
        ckpt = torch.load(tmp_path / name, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "model_kwargs" in ckpt
        assert set(ckpt["model_kwargs"]) == {"D", "L", "step_dim"}


def test_logs_schema(tmp_path):
    run_tiny_training(tmp_path, epochs_a=2)
    jsonl_lines = (tmp_path / "train_log.jsonl").read_text().splitlines()
    assert jsonl_lines
    for line in jsonl_lines:
        rec = json.loads(line)
        assert {"ts", "stage", "epoch", "step", "loss", "lr",
                "grad_norm", "examples_per_sec"} <= set(rec)
    csv_lines = (tmp_path / "train_log.csv").read_text().splitlines()
    assert csv_lines[0] == CSV_HEADER
    assert len(csv_lines) == 3  # header + 2 epoch rows


# ---------------------------------------------------------------------------
# Stage-B kwarg filtering (cross-unit contract C2)
# ---------------------------------------------------------------------------

OFFERED = {
    "epochs": 1, "N": 2, "B": 10.0, "beta": 0.01, "lr": 1e-4,
    "checkpoint_dir": "ckpts", "device": "cpu", "purist": False, "steps": 3,
    "grad_clip": 1.0, "seed": 0, "entropy_coef": 0.0, "eps_clip": 0.2,
}


def test_filter_kwargs_current_signature():
    # exact current signature of marc.train.stage_b.train_stage_b on main
    def fake_train_stage_b(policy, ref_policy, problems, alpha_bar, epochs=5,
                           N=8, B=10.0, beta=0.01, lr=1e-4,
                           checkpoint_dir="checkpoints/stage_b", device="cpu",
                           purist=False):
        return (policy, epochs, N)

    kwargs = filter_kwargs(fake_train_stage_b, OFFERED)
    for dropped in ("steps", "grad_clip", "seed", "entropy_coef", "eps_clip"):
        assert dropped not in kwargs
    fake_train_stage_b("p", "r", [], None, **kwargs)  # must not TypeError


def test_filter_kwargs_extended_signature():
    def fake_train_stage_b(policy, ref_policy, problems, alpha_bar, epochs=5,
                           N=8, B=10.0, beta=0.01, lr=1e-4,
                           checkpoint_dir="checkpoints/stage_b", device="cpu",
                           purist=False, steps=40, grad_clip=None):
        return steps

    kwargs = filter_kwargs(fake_train_stage_b, OFFERED)
    assert kwargs["steps"] == 3
    assert kwargs["grad_clip"] == 1.0
    assert fake_train_stage_b("p", "r", [], None, **kwargs) == 3


# ---------------------------------------------------------------------------
# Smoke CLI (cross-unit contract C4)
# ---------------------------------------------------------------------------

def test_smoke_cli(tmp_path):
    out_dir = tmp_path / "run"
    main(["--config", str(REPO_YAML), "--smoke", "--out-dir", str(out_dir)])
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "stage_b_final.pt").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "train_log.csv").exists()
    assert (out_dir / "smoke_data" / "manifest.json").exists()
    ckpt = torch.load(out_dir / "latest.pt", weights_only=False)
    assert ckpt["model_kwargs"] == {"D": 32, "L": 2, "step_dim": 16}
