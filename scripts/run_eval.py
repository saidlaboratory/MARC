#!/usr/bin/env python3
"""
P0 smoke eval: builds 5 sample 2x2 linear-equation graphs with known solutions,
runs DummySolver + Checker, and writes results/p0_smoke/metrics.json.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --n 20 --out results/my_run/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.runner import DummySolver, Problem, run_eval
from marc.graph.graph import FactorGraph
from marc.graph.schema import VariableNode, FactorNode, Edge


def make_sample_problems(n: int) -> list[Problem]:
    """Each problem: x + y = a, x - y = b, with solution x=(a+b)/2, y=(a-b)/2."""
    problems = []
    for i in range(n):
        a, b = 3 + i, 1 + i
        graph = FactorGraph(
            variables=[VariableNode("x"), VariableNode("y")],
            factors=[
                FactorNode("eq1", f"x+y-{a}"),
                FactorNode("eq2", f"x-y-{b}"),
            ],
            edges=[
                Edge("x", "eq1", 1), Edge("y", "eq1", 1),
                Edge("x", "eq2", 1), Edge("y", "eq2", -1),
            ],
        )
        solution = [(a + b) / 2, (a - b) / 2]
        problems.append(
            Problem(
                id=f"sample_{i:03d}",
                graph=graph,
                solution=solution,
                description=f"x+y={a}, x-y={b}",
            )
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 smoke eval with Checker")
    parser.add_argument("--n", type=int, default=5, help="Number of sample problems")
    parser.add_argument(
        "--out",
        default="results/p0_smoke/metrics.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1, help="Candidates per problem (pass@k)")
    args = parser.parse_args()

    problems = make_sample_problems(args.n)
    solver = DummySolver(seed=args.seed)
    metrics = run_eval(problems, solver=solver, n_samples=args.n_samples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote metrics to {out_path}")
    print(f"  solve_rate (pass@1) : {metrics['solve_rate']:.3f}")
    print(f"  pass_at_k (k={metrics['n_samples']})     : {metrics['pass_at_k']:.3f}")
    print(f"  generalization_gap  : {metrics['generalization_gap']:.3f}")
    print(f"  entrapment_rate     : {metrics['entrapment_rate']:.3f}")
    print(f"  perturbation_rob.   : {metrics['perturbation_robustness']:.3f}")


if __name__ == "__main__":
    main()
