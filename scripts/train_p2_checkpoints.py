#!/usr/bin/env python3
"""Train the checkpoints the P2 paper eval needs (small/CPU-friendly, real training).

Produces three checkpoints under ``checkpoints/`` (gitignored — regenerate locally):

  * ``denoiser_stage_a.pt``          — Stage-A DSM pretraining (reference policy).
  * ``denoiser_stage_b_standard.pt`` — Stage-B GRPO, terminal reward + energy shaping.
  * ``denoiser_stage_b_purist.pt``   — Stage-B GRPO, terminal reward only (purist=True).

Each checkpoint carries ``model_kwargs`` so :class:`marc.eval.solver.LearnedSolver`
reconstructs the exact architecture (see the ``a1df5a9`` checkpoint-loading fix).

Usage:
    python scripts/train_p2_checkpoints.py
    MARC_CKPT=checkpoints/denoiser_stage_b_standard.pt \\
    MARC_CKPT_PURIST=checkpoints/denoiser_stage_b_purist.pt \\
        python scripts/run_main_eval.py --solver learned
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.cas.engine import CASEngine
from marc.data.collate import collate_fn
from marc.data.dataset import MARCDataset
from marc.data.generator import ProblemGenerator
from marc.data.templates import LinearSystem2x2Template, LinearSystem3x3Template
from marc.diffusion.schedule import cosine_beta_schedule
from marc.graph.serialize import load_graph
from marc.model.denoiser import GraphDenoiser
from marc.train.stage_a import train_stage_a
from marc.train.stage_b import train_stage_b

MODEL_KWARGS = {"D": 128, "L": 4, "step_dim": 64}
CKPT_DIR = Path("checkpoints")


def _save(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_kwargs": MODEL_KWARGS}, path)
    print(f"  -> {path}")


def train_a(train_pairs, epochs: int = 15) -> GraphDenoiser:
    print(f"\n== Stage A: DSM pretraining ({len(train_pairs)} pairs, {epochs} epochs) ==")
    dataset = MARCDataset(train_pairs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    _, alpha_bar = cosine_beta_schedule(1000)
    denoiser = GraphDenoiser(**MODEL_KWARGS)
    denoiser = train_stage_a(
        denoiser, loader, alpha_bar, T=1000, epochs=epochs,
        checkpoint_dir=str(CKPT_DIR / "stage_a_epochs"), lr=1e-3,
    )
    _save(denoiser, CKPT_DIR / "denoiser_stage_a.pt")
    return denoiser


def _stage_b_problems(train_pairs):
    """(graph, solution, CASEngine) tuples for GRPO rollouts."""
    out = []
    for graph_path, _solution_path in train_pairs:
        graph = load_graph(graph_path)
        symbol_names = [v.id for v in graph.variables]
        cas = CASEngine(graph_path, symbol_names)
        out.append((graph, None, cas))
    return out


def train_b(stage_a_model: GraphDenoiser, problems, *, purist: bool, epochs: int = 3):
    label = "purist" if purist else "standard"
    print(f"\n== Stage B: GRPO fine-tuning ({label}, {len(problems)} problems, {epochs} epochs) ==")
    _, alpha_bar = cosine_beta_schedule(1000)
    policy = GraphDenoiser(**MODEL_KWARGS)
    policy.load_state_dict(copy.deepcopy(stage_a_model.state_dict()))
    ref_policy = GraphDenoiser(**MODEL_KWARGS)
    ref_policy.load_state_dict(copy.deepcopy(stage_a_model.state_dict()))

    policy = train_stage_b(
        policy, ref_policy, problems, alpha_bar,
        epochs=epochs, N=4, B=10.0, beta=0.01, lr=1e-4,
        checkpoint_dir=str(CKPT_DIR / f"stage_b_{label}_epochs"), purist=purist,
    )
    out_path = CKPT_DIR / f"denoiser_stage_b_{label}.pt"
    _save(policy, out_path)
    return out_path


def main() -> None:
    import os

    # Backward-compatible scale knobs (defaults reproduce the original smoke run).
    n_per = int(os.environ.get("MARC_N_PER", "70"))
    epochs_a = int(os.environ.get("MARC_EPOCHS_A", "15"))
    epochs_b = int(os.environ.get("MARC_EPOCHS_B", "3"))
    b_subset = int(os.environ.get("MARC_B_SUBSET", "16"))

    out_dir = Path("results/p2_main/train_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ProblemGenerator(
        templates=[LinearSystem2x2Template(), LinearSystem3x3Template()],
        split_ratio=0.85,
        seed=42,
    )
    print(f"Generating training problems (n_per_template={n_per}) ...")
    train_pairs, _test_pairs = gen.generate(n_per_template=n_per, output_dir=str(out_dir))
    print(f"  {len(train_pairs)} training pairs.")

    stage_a_model = train_a(train_pairs, epochs=epochs_a)

    # Stage B trains on a smaller subset — GRPO does N rollouts x steps per problem.
    b_problems = _stage_b_problems(train_pairs[:b_subset])
    std_path = train_b(stage_a_model, b_problems, purist=False, epochs=epochs_b)
    purist_path = train_b(stage_a_model, b_problems, purist=True, epochs=epochs_b)

    print("\n== done ==")
    print(f"MARC_CKPT={CKPT_DIR / 'denoiser_stage_a.pt'}")
    print(f"MARC_CKPT (standard Stage-B)={std_path}")
    print(f"MARC_CKPT_PURIST={purist_path}")


if __name__ == "__main__":
    main()
