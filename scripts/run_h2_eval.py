#!/usr/bin/env python3
"""P3 H2 eval — does inventing an auxiliary object help solving? (marc/eval/structure_eval.py)

Runs the fixed vs. structure model over the 3 toy families and writes:

  results/p3_h2/summary.json        — per-toy + overall metrics
  results/p3_h2/trajectories.jsonl  — one line per (toy, instance, model) run

Usage:
    python scripts/run_h2_eval.py
    python scripts/run_h2_eval.py --n 25 --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.structure_eval import run_h2_suite

OUT_DIR = Path("results/p3_h2")


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 H2 eval (fixed vs. structure model)")
    parser.add_argument("--n", type=int, default=25, help="instances per toy")
    parser.add_argument("--k", type=int, default=10, help="restarts per instance")
    parser.add_argument("--steps", type=int, default=300, help="refine() steps per restart")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== P3 H2 eval :: n={args.n} k={args.k} steps={args.steps} ==")
    summary, fixed_records, structure_records = run_h2_suite(
        n_instances=args.n, k=args.k, steps=args.steps, base_seed=args.base_seed,
    )

    for name, row in summary["toys"].items():
        print(
            f"[{name}] fixed={row['fixed_solve_rate']:.2f} "
            f"structure={row['structure_solve_rate']:.2f} "
            f"aux_usage={row['auxiliary_usage_rate']:.2f}"
        )
    print(
        f"[overall] fixed={summary['overall_fixed_solve_rate']:.2f} "
        f"structure={summary['overall_structure_solve_rate']:.2f} "
        f"aux_usage={summary['overall_auxiliary_usage_rate']:.2f}"
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"-> {summary_path}")

    traj_path = out_dir / "trajectories.jsonl"
    with traj_path.open("w") as fh:
        for record in fixed_records:
            fh.write(json.dumps(record.to_dict(model="fixed")) + "\n")
        for record in structure_records:
            fh.write(json.dumps(record.to_dict(model="structure")) + "\n")
    print(f"-> {traj_path}")


if __name__ == "__main__":
    main()
