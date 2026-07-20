"""Config-driven trainer for the scale run (Stage A DSM + optional Stage B GRPO).

The real training entry point behind ``scripts/train_scale.py``:

    python3 scripts/train_scale.py --config marc/configs/train/scale.yaml \\
        --out-dir checkpoints/scale_D512_L8 [--resume latest|PATH] \\
        [--device auto|cuda|mps|cpu] [--stage a|b|ab] [--smoke]

Adds what the P2 CPU script lacks: device resolution, seeding, bf16 AMP,
grad clipping, warmup+cosine LR schedule, atomic checkpoints with resume,
and structured JSONL/CSV logging. Checkpoints carry ``model_state_dict`` +
``model_kwargs`` so :class:`marc.eval.solver.LearnedSolver` loads them as-is.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from marc.data.collate import collate_fn
from marc.data.dataset import MARCDataset
from marc.data.generator import ProblemGenerator
from marc.diffusion.forward import corrupt
from marc.diffusion.schedule import cosine_beta_schedule
from marc.model.denoiser import GraphDenoiser
from marc.train.stage_b import train_stage_b

# NOTE: no imports from marc.train.stage_a — that file is being rewritten by
# another unit; the Stage-A loss math is reproduced in stage_a_loss_local.

DEFAULTS = {
    "model": {"D": 512, "L": 8, "step_dim": 64},
    "training": {
        "T": 1000,
        "epochs_A": 50,
        "epochs_B": 20,
        "batch_size": 32,
        "lr_A": 3.0e-4,
        "lr_B": 1.0e-4,
        "device": "auto",
        "seed": 42,
        "amp": "auto",
        "grad_clip": 1.0,
        "warmup_frac": 0.03,
        "num_workers": 2,
    },
    "data": {
        "n_train": 10000,  # TOTAL problems across all templates
        "n_test": 2000,
        "templates": [
            "LinearSystem2x2",
            "LinearSystem3x3",
            "TriangleDistance",
            "PointSlope",
            "BilinearSystem",
            "BilinearProduct",
            "QuadraticSystem",
            "CircleLine",
        ],
        "dir": "results/scale/train_data",
        "seed": 42,
    },
    "grpo": {
        "N": 8,
        "B": 10.0,
        "beta": 0.01,
        "eps_clip": 0.2,
        "steps": 40,
        "guidance_weight": 1.0,
        "problems_subset": 256,
    },
    "checkpointing": {"dir": "checkpoints/scale_D512_L8", "save_every_n_epochs": 5},
}

CSV_HEADER = "stage,epoch,loss,lr,grad_norm,examples_per_sec,wall_time"


# ---------------------------------------------------------------------------
# Config / environment
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str, cli_overrides: dict | None = None) -> dict:
    """yaml.safe_load deep-merged over DEFAULTS, then CLI overrides on top.

    A yaml missing the newer keys (seed/amp/grad_clip/...) still loads — every
    new key is defaulted in code, not required in the file.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _deep_merge(DEFAULTS, raw)
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)
    return cfg


def resolve_device(spec: str = "auto") -> str:
    """auto -> cuda if available, else mps, else cpu. Explicit spec passes through."""
    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_amp(spec, device: str) -> bool:
    # ponytail: bf16-or-off — no fp16/GradScaler path; fp32 on mps/cpu.
    if device != "cuda":
        return False
    if spec == "auto":
        return torch.cuda.is_bf16_supported()
    return bool(spec)


# ---------------------------------------------------------------------------
# Templates / data
# ---------------------------------------------------------------------------

def template_registry() -> dict:
    """Scan marc.data.templates for zero-arg *Template classes, keyed by .name.

    A module scan, so future template families auto-compose without edits here.
    """
    import marc.data.templates as templates_mod

    registry = {}
    for attr in dir(templates_mod):
        if not attr.endswith("Template"):
            continue
        cls = getattr(templates_mod, attr)
        if not isinstance(cls, type):
            continue
        try:
            instance = cls()
        except TypeError:
            continue  # needs constructor args; not a plain template
        key = getattr(instance, "name", None) or attr[: -len("Template")]
        registry[key] = instance
    return registry


def resolve_templates(names) -> list:
    reg = template_registry()
    out = []
    for n in names:
        if n in reg:
            out.append(reg[n])
        else:
            warnings.warn(f"unknown template {n!r}; known: {sorted(reg)}")
    if not out:
        raise ValueError(f"no valid templates in config (asked for {list(names)})")
    return out


def prepare_data(cfg: dict):
    """Generate (or reuse) the training set; returns (train_pairs, test_pairs).

    Writes {data.dir}/manifest.json; if a manifest already matches the config
    the dataset is reused and regeneration (CAS-verifying every sympy instance,
    expensive at 10k) is skipped — resume must not repay it.
    """
    dcfg = cfg["data"]
    templates = resolve_templates(dcfg["templates"])
    data_dir = dcfg["dir"]
    manifest_path = os.path.join(data_dir, "manifest.json")
    key = {
        "templates": [t.name for t in templates],
        "n_train": int(dcfg["n_train"]),
        "seed": int(dcfg["seed"]),
    }

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            manifest = None
        if manifest and all(manifest.get(k) == v for k, v in key.items()):
            train_pairs = [tuple(p) for p in manifest["train_pairs"]]
            test_pairs = [tuple(p) for p in manifest["test_pairs"]]
            if all(os.path.exists(p) for pair in train_pairs + test_pairs for p in pair):
                print(f"[data] manifest match — reusing {len(train_pairs)} train pairs from {data_dir}")
                return train_pairs, test_pairs
            print("[data] manifest matched but files are missing — regenerating")

    # n_train is the TOTAL target; the generator takes a per-template count.
    n_per = max(1, key["n_train"] // len(templates))
    print(f"[data] generating {n_per} problems x {len(templates)} templates into {data_dir} ...")
    gen = ProblemGenerator(templates, split_ratio=0.85, seed=key["seed"])
    train_pairs, test_pairs = gen.generate(n_per_template=n_per, output_dir=data_dir)

    manifest = {**key, "train_pairs": train_pairs, "test_pairs": test_pairs}
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh)
    os.replace(tmp, manifest_path)
    return train_pairs, test_pairs


def build_loader(train_pairs, cfg: dict, device: str) -> DataLoader:
    """Materialize the dataset in memory once (build_heterodata sympifies per
    __getitem__ — re-parsing 10k graphs every epoch would starve the GPU)."""
    dataset = MARCDataset(train_pairs)
    items = [dataset[i] for i in range(len(dataset))]
    return DataLoader(
        items,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )


# ---------------------------------------------------------------------------
# Schedule / logging / checkpoints
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR: linear warmup for warmup_steps, then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / denom)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class RunLogger:
    """Structured logs: per-step JSONL + per-epoch CSV (printed too). No tqdm."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "train_log.jsonl")
        self.csv_path = os.path.join(out_dir, "train_log.csv")
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "a") as fh:
                fh.write(CSV_HEADER + "\n")

    def step(self, **kw) -> None:
        record = {"ts": time.time(), **kw}
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def epoch(self, *, stage, epoch, loss, lr, grad_norm, examples_per_sec, wall_time) -> None:
        row = (
            f"{stage},{epoch},{loss:.6f},{lr:.6e},{grad_norm:.4f},"
            f"{examples_per_sec:.2f},{wall_time:.2f}"
        )
        with open(self.csv_path, "a") as fh:
            fh.write(row + "\n")
        print(row)


def save_checkpoint(path, *, epoch, global_step, stage, model, model_kwargs, config,
                    loss, wall_time, optimizer=None, scheduler=None) -> None:
    """Atomic (tmp + os.replace). Carries model_state_dict + model_kwargs so
    LearnedSolver reconstructs the exact architecture."""
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_kwargs": dict(model_kwargs),
        "config": config,
        "loss": loss,
        "wall_time": wall_time,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def stage_a_loss_local(model, data, solutions, alpha_bar, T: int, device: str):
    """DSM loss for one collated batch. THE SEAM (cross-unit contract C1):

    If the model advertises ``supports_per_graph_t`` AND the batch carries a
    per-node graph index, corrupt with a per-graph timestep vector and do ONE
    batched forward. Dispatch is attribute-based ONLY — on today's main a
    vector t does not raise (forward takes t.view(-1)[0]), it silently trains
    wrong, so try/except detection is forbidden.

    Otherwise: the per-graph loop, math copied from main's train_step_A
    (minus the optimizer; no import — that file is being rewritten).
    """
    vb = getattr(data["variable"], "batch", None)
    if vb is not None and getattr(model, "supports_per_graph_t", False):
        x0 = torch.cat([s.to(device) for s in solutions], dim=0)
        num_graphs = int(vb.max().item()) + 1
        t = torch.randint(1, T + 1, (num_graphs,), device=device)
        eps = torch.randn_like(x0)
        # corrupt already handles per-element t; index it out per node.
        data["variable"].x = corrupt(x0, t[vb], eps, alpha_bar)
        return F.mse_loss(model(data, t), eps)

    total = torch.tensor(0.0, device=device)
    try:
        graphs = data.to_data_list()
    except AttributeError:
        graphs = [data] * len(solutions)
    for graph, x0 in zip(graphs, solutions):
        x0 = x0.to(device)
        t = torch.randint(1, T + 1, (1,), device=device)
        eps = torch.randn_like(x0)
        graph["variable"].x = corrupt(x0, t, eps, alpha_bar)
        total = total + F.mse_loss(model(graph, t), eps)
    return total / max(len(solutions), 1)


def train_stage_a_scaled(model, loader, alpha_bar, cfg, device, out_dir, logger,
                         model_kwargs, resume_path=None):
    """Stage-A loop: AMP -> backward -> clip -> step -> scheduler -> log."""
    tr = cfg["training"]
    T, epochs, grad_clip = tr["T"], tr["epochs_A"], tr["grad_clip"]
    save_every = cfg["checkpointing"]["save_every_n_epochs"]

    model = model.to(device)
    alpha_bar = alpha_bar.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr_A"])
    total_steps = max(1, len(loader) * epochs)
    warmup_steps = int(tr["warmup_frac"] * total_steps)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    amp_on = _resolve_amp(tr["amp"], device)
    print(f"[amp] {'bf16 autocast' if amp_on else 'fp32'} (amp={tr['amp']!r}, device={device})")

    start_epoch, global_step, best_loss = 0, 0, float("inf")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("stage") not in (None, "a"):
            warnings.warn(f"resuming Stage A from a stage={ckpt.get('stage')!r} checkpoint")
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        # ponytail: epoch-granular resume; best_loss restarts from the resumed
        # epoch's loss, so best.pt may briefly regress after a resume.
        best_loss = float(ckpt.get("loss", float("inf")))
        print(f"[resume] {resume_path} -> continuing at epoch {start_epoch + 1}")

    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        losses, grad_norms, n_examples = [], [], 0
        lr_now = optimizer.param_groups[0]["lr"]

        for data, solutions in loader:
            step_start = time.time()
            data = data.to(device)
            optimizer.zero_grad()
            if amp_on:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = stage_a_loss_local(model, data, solutions, alpha_bar, T, device)
            else:
                loss = stage_a_loss_local(model, data, solutions, alpha_bar, T, device)
            loss.backward()
            gn = float(clip_grad_norm_(model.parameters(), grad_clip))
            lr_now = optimizer.param_groups[0]["lr"]  # lr actually used this step
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_val = float(loss.item())
            losses.append(loss_val)
            grad_norms.append(gn)
            n_examples += len(solutions)
            logger.step(
                stage="a", epoch=epoch + 1, step=global_step, loss=loss_val,
                lr=lr_now, grad_norm=gn,
                examples_per_sec=len(solutions) / max(time.time() - step_start, 1e-9),
            )

        if not losses:
            warnings.warn("empty dataloader — nothing trained this epoch")
            break
        epoch_loss = sum(losses) / len(losses)
        wall = time.time() - epoch_start
        logger.epoch(
            stage="a", epoch=epoch + 1, loss=epoch_loss, lr=lr_now,
            grad_norm=sum(grad_norms) / len(grad_norms),
            examples_per_sec=n_examples / max(wall, 1e-9), wall_time=wall,
        )

        ckpt_kw = dict(
            epoch=epoch + 1, global_step=global_step, stage="a", model=model,
            model_kwargs=model_kwargs, config=cfg, loss=epoch_loss,
            wall_time=time.time() - run_start, optimizer=optimizer, scheduler=scheduler,
        )
        save_checkpoint(os.path.join(out_dir, "latest.pt"), **ckpt_kw)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(out_dir, f"epoch_{epoch + 1:04d}.pt"), **ckpt_kw)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(os.path.join(out_dir, "best.pt"), **ckpt_kw)

    return model


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

def filter_kwargs(fn, offered: dict) -> dict:
    """Cross-unit contract C2: pass only kwargs present in fn's signature, so
    this works against today's train_stage_b and auto-gains knobs (steps,
    grad_clip, ...) if the Stage-B rewrite lands."""
    sig = inspect.signature(fn)
    return {k: v for k, v in offered.items() if k in sig.parameters}


def run_stage_b(model, train_pairs, alpha_bar, cfg, device, out_dir, model_kwargs):
    """GRPO fine-tune from the Stage-A weights; saves stage_b_final.pt."""
    from marc.cas.engine import CASEngine
    from marc.graph.serialize import load_graph

    g, tr = cfg["grpo"], cfg["training"]
    subset = train_pairs[: int(g["problems_subset"])]
    print(f"[stage b] building {len(subset)} problems (CAS engines) ...")
    problems = []
    for graph_path, _solution_path in subset:
        graph = load_graph(graph_path)
        ids = [v.id for v in graph.variables]
        problems.append((graph, None, CASEngine(graph_path, ids)))

    ref_policy = GraphDenoiser(**model_kwargs)
    ref_policy.load_state_dict(copy.deepcopy(model.state_dict()))

    offered = {
        "epochs": tr["epochs_B"],
        "N": g["N"],
        "B": g["B"],
        "beta": g["beta"],
        "lr": tr["lr_B"],
        "checkpoint_dir": os.path.join(out_dir, "stage_b_epochs"),
        "device": device,
        "purist": False,
        "steps": g["steps"],
        "grad_clip": tr["grad_clip"],
        "seed": tr["seed"],
        "entropy_coef": g.get("entropy_coef", 0.0),
        "eps_clip": g["eps_clip"],
    }
    start = time.time()
    model = train_stage_b(model, ref_policy, problems, alpha_bar,
                          **filter_kwargs(train_stage_b, offered))
    save_checkpoint(
        os.path.join(out_dir, "stage_b_final.pt"),
        epoch=tr["epochs_B"], global_step=0, stage="b", model=model,
        model_kwargs=model_kwargs, config=cfg, loss=float("nan"),
        wall_time=time.time() - start,
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _apply_smoke(cfg: dict, out_dir: str) -> dict:
    """Tiny post-load overrides; the whole run must finish in <2 minutes."""
    return _deep_merge(cfg, {
        "model": {"D": 32, "L": 2, "step_dim": 16},
        "training": {"epochs_A": 2, "epochs_B": 1, "batch_size": 4,
                     "device": "cpu", "num_workers": 0},
        # ponytail: smoke data goes under out_dir so it never pollutes (or
        # invalidates the manifest of) the real dataset cache.
        "data": {"n_train": 12, "n_test": 4, "templates": ["LinearSystem2x2"],
                 "dir": os.path.join(out_dir, "smoke_data")},
        "grpo": {"N": 2, "steps": 3, "problems_subset": 2},
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven Stage-A (DSM) / Stage-B (GRPO) trainer for the scale run.",
        epilog=(
            "NOTE: on current main, Stage-B GRPO carries a known log-prob bug; "
            "recommend --stage a unless the Stage-B fix has merged."
        ),
    )
    parser.add_argument("--config", required=True, help="path to a train yaml (e.g. marc/configs/train/scale.yaml)")
    parser.add_argument("--out-dir", required=True, help="checkpoint + log directory")
    parser.add_argument("--resume", default=None, help='"latest" (out-dir/latest.pt) or a checkpoint path')
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"],
                        help="override training.device from the config")
    parser.add_argument("--stage", default="ab", choices=["a", "b", "ab"])
    parser.add_argument("--smoke", action="store_true", help="tiny <2 min end-to-end run")
    args = parser.parse_args(argv)

    cli_overrides = {"training": {"device": args.device}} if args.device else None
    cfg = load_config(args.config, cli_overrides)
    if args.smoke:
        cfg = _apply_smoke(cfg, args.out_dir)

    device = resolve_device(cfg["training"]["device"])
    set_seed(cfg["training"]["seed"])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger = RunLogger(out_dir)
    print(f"[trainer] device={device} stage={args.stage} out_dir={out_dir}")

    resume_path = None
    if args.resume:
        resume_path = os.path.join(out_dir, "latest.pt") if args.resume == "latest" else args.resume
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    train_pairs, _test_pairs = prepare_data(cfg)
    _, alpha_bar = cosine_beta_schedule(cfg["training"]["T"])
    model_kwargs = {"D": cfg["model"]["D"], "L": cfg["model"]["L"],
                    "step_dim": cfg["model"]["step_dim"]}
    model = GraphDenoiser(**model_kwargs)

    if args.stage in ("a", "ab"):
        loader = build_loader(train_pairs, cfg, device)
        model = train_stage_a_scaled(model, loader, alpha_bar, cfg, device, out_dir,
                                     logger, model_kwargs, resume_path=resume_path)
    else:  # stage b only — needs Stage-A weights from somewhere
        init_path = resume_path or os.path.join(out_dir, "latest.pt")
        if os.path.exists(init_path):
            ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            print(f"[stage b] initialized from {init_path}")
        else:
            warnings.warn("Stage B starting from RANDOM weights (no --resume and no latest.pt)")
        model = model.to(device)

    if args.stage in ("b", "ab"):
        run_stage_b(model, train_pairs, alpha_bar.to(device), cfg, device, out_dir,
                    model_kwargs)

    print(f"[trainer] done — checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
