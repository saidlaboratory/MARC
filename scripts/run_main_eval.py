#!/usr/bin/env python3
"""P2 main paper eval — run all suites + ablations into ``results/p2_main/``.

Suites (TECHNICAL_GUIDE §11):
  * generalization gap      -> results/p2_main/generalization.json
  * perturbation sweep      -> results/p2_main/perturbation.json
  * length extrapolation    -> results/p2_main/length_extrapolation.json

Ablations (one JSON each, checkpoint-gated where they need the learned model):
  * noise on/off            -> results/p2_main/ablation_noise.json
  * guidance weight         -> results/p2_main/ablation_guidance.json
  * purist reward           -> results/p2_main/ablation_purist.json

The default solver is the real ``refine`` baseline, which runs today with no GPU and
no checkpoint. Swap in Quang's Stage-B weights the moment they land:

    MARC_CKPT=/path/to/denoiser.pt python scripts/run_main_eval.py --solver learned

The guidance/purist ablations write a ``status: "skipped"`` record (not a failure)
until their checkpoints exist, so the pipeline is green end-to-end from day one.

Usage:
    python scripts/run_main_eval.py
    python scripts/run_main_eval.py --solver learned --k 8
    python scripts/run_main_eval.py --skip-ablations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.ablations import guidance_ablation, purist_ablation
from marc.eval.ablations import noise_ablation
from marc.eval.paper.suites import (
    run_generalization_gap,
    run_length_extrapolation,
    run_perturbation,
)
from marc.eval.solver import load_solver

OUT_DIR = Path("results/p2_main")


def _write(name: str, payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 main paper eval (suites + ablations)")
    parser.add_argument("--solver", default="refine", help="refine | dummy | learned")
    parser.add_argument("--k", type=int, default=4, help="candidates per problem (pass@k)")
    parser.add_argument("--n", type=int, default=25, help="problems per split/length bucket")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--noise-graphs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    solver = load_solver(args.solver)
    print(f"== P2 main eval :: solver={args.solver} k={args.k} n={args.n} ==")

    # ---- suites --------------------------------------------------------------
    gen = run_generalization_gap(solver, n_id=args.n, n_ho=args.n, k=args.k)
    p = _write("generalization.json", gen, out_dir)
    print(f"[generalization] gap={gen['generalization_gap']:.3f} -> {p}")

    pert = run_perturbation(solver, n_id=args.n, n_ho=args.n, k=args.k)
    p = _write("perturbation.json", pert, out_dir)
    print(f"[perturbation]   {len(pert['deltas'])} deltas -> {p}")

    length = run_length_extrapolation(solver, n=args.n, k=args.k)
    p = _write("length_extrapolation.json", length, out_dir)
    print(f"[length]         {len(length['lengths'])} lengths -> {p}")

    if args.skip_ablations:
        print("== skipping ablations (--skip-ablations) ==")
        return

    # ---- ablations -----------------------------------------------------------
    noise_summary = noise_ablation.run_ablation(n_graphs=args.noise_graphs)
    noise_serialisable = {k: v for k, v in noise_summary.items() if not k.startswith("_")}
    p = _write("ablation_noise.json", noise_serialisable, out_dir)
    print(
        f"[noise]          reduction={noise_summary['entrapment_reduction_mean']:.3f} -> {p}"
    )

    guid = guidance_ablation.run_ablation(k=args.k, n_ho=args.n)
    p = _write("ablation_guidance.json", guid, out_dir)
    print(f"[guidance]       status={guid['status']} -> {p}")

    pur = purist_ablation.run_ablation(n=args.n, k=args.k)
    p = _write("ablation_purist.json", pur, out_dir)
    print(f"[purist]         status={pur['status']} -> {p}")

    print(f"== done. results in {out_dir}/ ==")


if __name__ == "__main__":
    main()
