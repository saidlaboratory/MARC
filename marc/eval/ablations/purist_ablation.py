"""Purist-reward ablation (TECHNICAL_GUIDE §7, §11).

Compares two Stage-B GRPO checkpoints: the **standard** reward (terminal checker
reward + per-step energy shaping) vs. the **purist** reward (terminal checker reward
only — the ``purist=True`` arm in :mod:`marc.train.stage_b`). The hypothesis is that
energy shaping helps optimisation without changing what's ultimately verified.

This is a *training* ablation: it needs two separately-trained checkpoints, so it is
**checkpoint-gated** on both ``MARC_CKPT`` (standard) and ``MARC_CKPT_PURIST``. Without
both it returns a ``status: "skipped"`` record. Each arm is evaluated on the
generalization-gap suite so the JSON carries solve rate, gap, and overall rate.

Run:
    MARC_CKPT=std.pt MARC_CKPT_PURIST=purist.pt \\
        python -m marc.eval.ablations.purist_ablation
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from marc.eval.paper.suites import run_generalization_gap

DEFAULT_OUT = "results/p2_main/ablation_purist.json"


def _skip(reason: str) -> dict:
    return {
        "ablation": "purist_reward",
        "status": "skipped",
        "reason": reason,
        "hint": (
            "Set MARC_CKPT (standard reward) and MARC_CKPT_PURIST (terminal-only "
            "reward) to Quang's two Stage-B checkpoints, and install torch_geometric."
        ),
    }


def _eval_checkpoint(checkpoint: str, *, n: int, k: int) -> dict:
    """Evaluate one checkpoint on the generalization-gap suite."""
    from marc.eval.solver import LearnedSolver

    solver = LearnedSolver(checkpoint=checkpoint)
    metrics = run_generalization_gap(solver, n_id=n, n_ho=n, k=k)
    return {
        "checkpoint": checkpoint,
        "in_distribution_solve_rate": metrics["splits"]["in_distribution"]["solve_rate"],
        "held_out_solve_rate": metrics["splits"]["held_out_structure"]["solve_rate"],
        "generalization_gap": metrics["generalization_gap"],
        "overall_solve_rate": metrics["overall_solve_rate"],
    }


def run_ablation(
    *,
    standard: str | None = None,
    purist: str | None = None,
    n: int = 25,
    k: int = 4,
) -> dict:
    """Evaluate standard vs. purist checkpoints; skip cleanly if either is missing."""
    standard = standard or os.environ.get("MARC_CKPT")
    purist = purist or os.environ.get("MARC_CKPT_PURIST")
    if not standard:
        return _skip("no standard checkpoint (MARC_CKPT unset)")
    if not purist:
        return _skip("no purist checkpoint (MARC_CKPT_PURIST unset)")

    try:
        std_arm = _eval_checkpoint(standard, n=n, k=k)
        purist_arm = _eval_checkpoint(purist, n=n, k=k)
    except Exception as exc:  # torch_geometric / checkpoint issues
        return _skip(f"learned solver failed at runtime: {exc}")

    delta = std_arm["overall_solve_rate"] - purist_arm["overall_solve_rate"]
    return {
        "ablation": "purist_reward",
        "status": "ok",
        "k": k,
        "standard": std_arm,
        "purist": purist_arm,
        "shaping_gain_overall_solve_rate": delta,
        "shaping_helps": delta > 0,
    }


def write_outputs(summary: dict, out: str | Path = DEFAULT_OUT) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Purist-reward ablation (Stage-B)")
    parser.add_argument("--standard", default=None, help="defaults to MARC_CKPT")
    parser.add_argument("--purist", default=None, help="defaults to MARC_CKPT_PURIST")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = run_ablation(
        standard=args.standard, purist=args.purist, n=args.n, k=args.k
    )
    path = write_outputs(summary, args.out)
    print(f"[purist] status={summary['status']} -> {path}")
    if summary["status"] == "skipped":
        print(f"[purist] skipped: {summary['reason']}")


if __name__ == "__main__":
    main()
