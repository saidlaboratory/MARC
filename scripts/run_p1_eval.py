#!/usr/bin/env python3
"""P1 baseline eval: in-distribution + held-out-structure splits → metrics.json.

Runs the real energy-gradient refinement solver (or Davin's learned ``solve()`` via
``--solver davin`` / ``MARC_SOLVER``) over both structural splits and writes
``results/p1_baselines/metrics.json`` with §11 capability metrics and the
generalization gap (H1).

Usage:
    python scripts/run_p1_eval.py
    python scripts/run_p1_eval.py --solver davin --k 8
    MARC_SOLVE_PATH=marc.diffusion.solve:solve python scripts/run_p1_eval.py --solver davin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.problems import held_out_structure, in_distribution
from marc.eval.runner import run_split_eval
from marc.eval.solver import load_solver


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 baseline eval (split metrics)")
    parser.add_argument("--solver", default="refine", help="refine | dummy | davin")
    parser.add_argument("--n-id", type=int, default=25, help="# in-distribution problems")
    parser.add_argument("--n-ho", type=int, default=25, help="# held-out-structure problems")
    parser.add_argument("--k", type=int, default=4, help="candidates per problem (pass@k)")
    parser.add_argument("--out", default="results/p1_baselines/metrics.json")
    args = parser.parse_args()

    solver = load_solver(args.solver)
    id_problems = in_distribution(n=args.n_id)
    ho_problems = held_out_structure(n=args.n_ho)

    metrics = run_split_eval(
        id_problems,
        ho_problems,
        solver=solver,
        n_samples=args.k,
        solver_name=args.solver,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    idm = metrics["splits"]["in_distribution"]
    hom = metrics["splits"]["held_out_structure"]
    print(f"Wrote baseline metrics to {out_path}")
    print(f"  solver               : {metrics['solver']}")
    print(f"  in-distribution      : solve_rate={idm['solve_rate']:.3f}  pass@{args.k}={idm['pass_at_k']:.3f}")
    print(f"  held-out-structure   : solve_rate={hom['solve_rate']:.3f}  pass@{args.k}={hom['pass_at_k']:.3f}")
    print(f"  generalization_gap   : {metrics['generalization_gap']:.3f}")
    print(f"  overall_solve_rate   : {metrics['overall_solve_rate']:.3f}")


if __name__ == "__main__":
    main()
