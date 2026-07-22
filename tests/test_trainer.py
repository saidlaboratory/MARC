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
from marc.model.denoiser import GraphDenoiser
from marc.train import stage_a
from marc.train.trainer import (
    CSV_HEADER,
    DEFAULTS,
    RunLogger,
    load_config,
    main,
    resolve_device,
    resolve_templates,
    stage_a_loss_local,
    template_registry,
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


def run_tiny_training(out_dir, epochs_a=2, model=None, resume_path=None, ema=True):
    cfg = tiny_cfg(epochs_a)
    cfg["training"]["ema"] = ema
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


def test_registry_includes_instance_registered_families():
    reg = template_registry()
    assert reg["CoupledChain4"].n == 4
    assert reg["CoupledChain6"].n == 6
    aux = sorted(k for k in reg if k.startswith("AuxRequired_"))
    assert "AuxRequired_offset" in aux and len(aux) >= 3
    resolved = resolve_templates(["CoupledChain4", "CoupledChain6", "AuxRequired_offset"])
    assert [t.name for t in resolved] == ["CoupledChain4", "CoupledChain6", "AuxRequired_offset"]
    # unknown names still warn+skip alongside the new families
    with pytest.warns(UserWarning, match="unknown template 'Bogus'"):
        resolved = resolve_templates(["CoupledChain4", "Bogus"])
    assert [t.name for t in resolved] == ["CoupledChain4"]


def test_registry_point_chain_families():
    from marc.cas.checker import Checker

    reg = template_registry()
    for k in (2, 3, 4):
        tmpl = reg[f"PointChain{k}"]
        assert tmpl.k == k
        graph, sol = tmpl.generate(seed=7)
        assert len(sol) == 2 * k
        assert set(sol) == {v.id for v in graph.variables}
        # the stored solution is exact — the checker gate accepts it as-is
        assert Checker().verify(graph, [sol[v.id] for v in graph.variables]).accepted
    resolved = resolve_templates(["PointChain2", "PointChain4"])
    assert [t.name for t in resolved] == ["PointChain2", "PointChain4"]


def test_scale_geo_yaml_loads_and_templates_resolve():
    cfg = load_config(str(REPO_YAML.parent / "scale_geo.yaml"))
    names = cfg["data"]["templates"]
    assert names[-3:] == ["PointChain2", "PointChain3", "PointChain4"]
    # different template mix => own data dir, so the live scale.yaml run's
    # manifest/cache is never invalidated
    assert cfg["data"]["dir"] != DEFAULTS["data"]["dir"]
    assert [t.name for t in resolve_templates(names)] == names


def test_scale_yaml_loads_and_templates_resolve():
    cfg = load_config(str(REPO_YAML))
    names = cfg["data"]["templates"]
    assert "CoupledChain4" in names and "CoupledChain6" in names
    # aux-required trains the structure policy, not the value denoiser
    assert not any(n.startswith("AuxRequired") for n in names)
    assert [t.name for t in resolve_templates(names)] == names  # all resolve, order kept
    assert cfg["training"]["ema"] is True
    assert cfg["training"]["ema_decay"] == 0.999


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


def test_trainer_batched_loss_matches_canonical_stage_a_loss():
    # Import parity: the trainer's loss IS marc.train.stage_a.stage_a_loss —
    # per-graph MSE then mean over graphs. Unequal graph sizes so the pre-#57
    # flat node-mean would give a different number.
    torch.manual_seed(0)
    items = [make_item(n_vars=2), make_item(n_vars=4), make_item(n_vars=2)]
    data, solutions = collate_fn(items)
    _, alpha_bar = cosine_beta_schedule(100)
    model = GraphDenoiser(D=16, L=1, step_dim=8)
    assert model.supports_per_graph_t
    t = torch.tensor([7, 42, 99])
    eps = torch.randn(sum(s.size(0) for s in solutions), 1)

    kw = dict(t=t, eps=eps)
    l_trainer = stage_a_loss_local(model, copy.deepcopy(data), solutions, alpha_bar, 100, "cpu", **kw)
    l_canon = stage_a.stage_a_loss(model, copy.deepcopy(data), solutions, alpha_bar, 100, "cpu", **kw)
    assert torch.allclose(l_trainer, l_canon)
    # batched path == per-graph loop on the same fixed (t, eps)
    l_loop = stage_a.stage_a_loss(
        model, copy.deepcopy(data), solutions, alpha_bar, 100, "cpu", batched=False, **kw
    )
    assert torch.allclose(l_trainer, l_loop, rtol=1e-4, atol=1e-5)


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
# EMA
# ---------------------------------------------------------------------------

def test_ema_in_checkpoint_and_diverges_from_raw(tmp_path):
    run_tiny_training(tmp_path, epochs_a=2)
    ckpt = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert "ema_state_dict" in ckpt
    raw, ema = ckpt["model_state_dict"], ckpt["ema_state_dict"]
    assert set(raw) == set(ema)
    # after 4 optimizer steps the 0.999-decay shadow lags the raw weights
    assert any(not torch.allclose(raw[k], ema[k]) for k in raw)
    assert "ema_state_dict" in torch.load(tmp_path / "best.pt", weights_only=False)


def test_ema_disabled_omits_checkpoint_key(tmp_path):
    run_tiny_training(tmp_path, epochs_a=1, ema=False)
    for name in ("latest.pt", "best.pt"):
        assert "ema_state_dict" not in torch.load(tmp_path / name, weights_only=False)


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
    assert ckpt["model_kwargs"] == {"D": 32, "L": 2, "step_dim": 16, "var_attn": False}
    assert "ema_state_dict" in ckpt  # EMA defaults on; smoke checkpoint carries it
