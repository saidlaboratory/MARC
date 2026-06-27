"""Guidance-weight ablation for the learned solver (TECHNICAL_GUIDE §5, §11).

Sweeps the CAS-guidance weight ``w`` in the diffusion ``solve()`` and records the
held-out-structure solve rate at each ``w``. Guidance only exists for the *learned*
solver (the gradient baseline has no denoiser to steer), so this ablation is
**checkpoint-gated**: without ``torch_geometric`` and a Stage-B checkpoint
(``MARC_CKPT``) it returns a clearly-marked ``status: "skipped"`` record rather than
fabricating numbers. Drop Quang's checkpoint in and re-run to populate it.

Run:  ``MARC_CKPT=/path/to/denoiser.pt python -m marc.eval.ablations.guidance_ablation``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from marc.eval.metrics import solve_rate
from marc.eval.problems import held_out_structure
from marc.eval.runner import run_eval

DEFAULT_OUT = "results/p2_main/ablation_guidance.json"
DEFAULT_WEIGHTS = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0]


def _skip(reason: str, weights: List[float]) -> dict:
    return {
        "ablation": "guidance",
        "status": "skipped",
        "reason": reason,
        "weights": weights,
        "hint": "Set MARC_CKPT to Quang's Stage-B checkpoint and install torch_geometric.",
    }


def run_ablation(
    weights: List[float] | None = None,
    *,
    checkpoint: str | None = None,
    n_ho: int = 25,
    k: int = 4,
) -> dict:
    """Sweep guidance weight on the learned solver; skip cleanly if it can't run."""
    weights = weights or DEFAULT_WEIGHTS
    checkpoint = checkpoint or os.environ.get("MARC_CKPT")
    if not checkpoint:
        return _skip("no checkpoint (MARC_CKPT unset)", weights)

    try:
        from marc.eval.solver import LearnedSolver
    except Exception as exc:  # pragma: no cover - import guard
        return _skip(f"learned solver import failed: {exc}", weights)

    problems = held_out_structure(n=n_ho)
    rows: List[dict] = []
    try:
        for w in weights:
            solver = LearnedSolver(checkpoint=checkpoint, guidance_weight=w)
            metrics = run_eval(problems, solver=solver, n_samples=k)
            rows.append(
                {
                    "guidance_weight": w,
                    "solve_rate": metrics["solve_rate"],
                    "pass_at_k": metrics["pass_at_k"],
                }
            )
    except Exception as exc:  # torch_geometric / checkpoint shape mismatch, etc.
        return _skip(f"learned solver failed at runtime: {exc}", weights)

    best = max(rows, key=lambda r: r["solve_rate"]) if rows else None
    return {
        "ablation": "guidance",
        "status": "ok",
        "checkpoint": checkpoint,
        "k": k,
        "weights": weights,
        "sweep": rows,
        "best_weight": best["guidance_weight"] if best else None,
        "guidance_helps": bool(rows) and rows[0]["solve_rate"] < best["solve_rate"],
    }


def write_outputs(summary: dict, out: str | Path = DEFAULT_OUT) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Guidance-weight ablation (learned solver)")
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--checkpoint", default=None, help="defaults to MARC_CKPT")
    parser.add_argument("--n-ho", type=int, default=25)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = run_ablation(
        args.weights, checkpoint=args.checkpoint, n_ho=args.n_ho, k=args.k
    )
    path = write_outputs(summary, args.out)
    print(f"[guidance] status={summary['status']} -> {path}")
    if summary["status"] == "skipped":
        print(f"[guidance] skipped: {summary['reason']}")


if __name__ == "__main__":
    main()
