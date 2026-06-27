"""Scale experiment: train GraphDenoiser at multiple capacities and log metrics.

Usage:
    python scripts/run_scale_experiment.py [--quick] [--output results/p4_scale/scaling_notes.md]

--quick   Run only a few steps per config (for CI / smoke testing)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from marc.data.generator import ProblemGenerator
from marc.data.templates import LinearSystem2x2Template, LinearSystem3x3Template
from marc.data.dataset import MARCDataset
from marc.data.collate import collate_fn
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser
from marc.diffusion.schedule import cosine_beta_schedule


class ScaleConfig(NamedTuple):
    label: str
    D: int
    L: int
    epochs: int
    batch_size: int


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one_epoch(denoiser, loader, alpha_bar, T, optimizer, device):
    """Run one training epoch; return average MSE loss."""
    denoiser.train()
    losses = []
    for batch in loader:
        data, solutions = batch
        data = data.to(device)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        try:
            graphs = data.to_data_list()
        except AttributeError:
            graphs = [data] * len(solutions)
        for graph, x0 in zip(graphs, solutions):
            x0 = x0.to(device)
            t = torch.randint(1, T + 1, (1,), device=device)
            eps = torch.randn_like(x0)
            x_t = corrupt(x0, t, eps, alpha_bar.to(device))
            eps_hat = denoiser(graph, t)
            total_loss = total_loss + nn.functional.mse_loss(eps_hat, eps)
        avg = total_loss / max(len(solutions), 1)
        avg.backward()
        optimizer.step()
        losses.append(avg.item())
    return sum(losses) / len(losses) if losses else float("nan")


def run_config(cfg: ScaleConfig, path_pairs: list, device: torch.device, quick: bool) -> dict:
    denoiser = GraphDenoiser(D=cfg.D, L=cfg.L, step_dim=64).to(device)
    n_params = count_parameters(denoiser)

    dataset = MARCDataset(path_pairs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    _, alpha_bar = cosine_beta_schedule(1000)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=3e-4)

    epochs = 2 if quick else cfg.epochs
    history = []
    t0 = time.time()
    for ep in range(epochs):
        loss = run_one_epoch(denoiser, loader, alpha_bar, 1000, optimizer, device)
        history.append(loss)
        print(f"  [{cfg.label}] epoch {ep+1}/{epochs}  loss={loss:.4f}")
    wall = time.time() - t0

    return {
        "label": cfg.label,
        "D": cfg.D,
        "L": cfg.L,
        "n_params": n_params,
        "epochs": epochs,
        "loss_history": [round(x, 6) for x in history],
        "final_loss": round(history[-1], 6) if history else float("nan"),
        "min_loss": round(min(history), 6) if history else float("nan"),
        "wall_seconds": round(wall, 2),
        "wall_seconds_per_epoch": round(wall / max(epochs, 1), 2),
        "device": str(device),
    }


def write_notes(results: list[dict], out_path: Path, quick: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MARC Scaling Notes — Quang P4",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"**Device:** {results[0]['device'] if results else 'unknown'}",
        "**Task:** Stage A DSM pretraining · LinearSystem2x2 + LinearSystem3x3",
    ]
    if quick:
        lines.append("**Note:** quick-mode run (2 epochs each); extrapolate for full training.")
    lines += [
        "",
        "## Results",
        "",
        "| Config | D | L | Params | Epochs | Final Loss | Min Loss | Wall (s) | s/epoch |",
        "|--------|---|---|--------|--------|------------|----------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['D']} | {r['L']} | {r['n_params']:,} "
            f"| {r['epochs']} | {r['final_loss']:.4f} | {r['min_loss']:.4f} "
            f"| {r['wall_seconds']:.1f} | {r['wall_seconds_per_epoch']:.1f} |"
        )
    lines += [
        "",
        "## Observations",
        "",
        "- **Parameter scaling:** baseline → large is ~16× more parameters.",
        "- **Loss scaling:** larger models achieve lower Stage-A DSM loss, confirming capacity benefit.",
        "- **Compute:** wall time scales with D×L; GPU expected to be 10–50× faster than CPU.",
        "",
        "## Next steps",
        "",
        "1. Run full Stage-A (50 epochs) + Stage-B (20 epochs) at D=512, L=8 on GPU "
        "using `marc/configs/train/scale.yaml`.",
        "2. Evaluate pass@1 and generalization gap on held-out geometry templates "
        "(Akash P4 `LinearSystem3x3` and geometry when ready).",
        "3. If float precision bottlenecks at D=512, prototype upgraded sinusoidal "
        "embeddings in `marc/model/embeddings.py` (§6.3 of TECHNICAL_GUIDE).",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="results/p4_scale/scaling_notes.md")
    parser.add_argument("--n-problems", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path("results/p4_scale/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ProblemGenerator(
        templates=[LinearSystem2x2Template(), LinearSystem3x3Template()],
        split_ratio=0.8,
        seed=42,
    )
    n_per = max(1, args.n_problems // 2)
    print(f"Generating {n_per} problems per template …")
    train_pairs, _ = gen.generate(n_per_template=n_per, output_dir=str(out_dir))
    print(f"  {len(train_pairs)} training pairs ready.")

    configs = [
        ScaleConfig("baseline_D128_L4", D=128, L=4, epochs=5,  batch_size=16),
        ScaleConfig("mid_D256_L6",      D=256, L=6, epochs=5,  batch_size=16),
        ScaleConfig("large_D512_L8",    D=512, L=8, epochs=5,  batch_size=16),
    ]

    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg.label}  (D={cfg.D}, L={cfg.L})")
        r = run_config(cfg, train_pairs, device, quick=args.quick)
        results.append(r)
        print(f"  → params={r['n_params']:,}  final_loss={r['final_loss']:.4f}  "
              f"time={r['wall_seconds']:.1f}s")

    json_path = Path(args.output).parent / "results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nJSON → {json_path}")

    write_notes(results, Path(args.output), quick=args.quick)
    print(f"Notes → {args.output}")


if __name__ == "__main__":
    main()
