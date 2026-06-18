#!/usr/bin/env python3
"""
P0 smoke eval: runs DummySolver on 5 fake problems and writes
results/p0_smoke/metrics.json.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --n 20 --out results/my_run/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.runner import DummySolver, Problem, run_eval


def make_fake_problems(n: int) -> list[Problem]:
    return [
        Problem(
            id=f"fake_{i:03d}",
            description=f"Fake problem {i}: solve for x in {i}x = {i * 3}",
            metadata={"difficulty": i % 3 + 1},
        )
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 smoke eval with DummySolver")
    parser.add_argument("--n", type=int, default=5, help="Number of fake problems")
    parser.add_argument(
        "--out",
        default="results/p0_smoke/metrics.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problems = make_fake_problems(args.n)
    solver = DummySolver(seed=args.seed)
    metrics = run_eval(problems, solver=solver)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote metrics to {out_path}")
    print(f"  solve_rate          : {metrics['solve_rate']:.3f}")
    print(f"  generalization_gap  : {metrics['generalization_gap']:.3f}")
    print(f"  entrapment_rate     : {metrics['entrapment_rate']:.3f}")
    print(f"  perturbation_rob.   : {metrics['perturbation_robustness']:.3f}")


if __name__ == "__main__":
    main()
