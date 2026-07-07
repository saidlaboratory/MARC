#!/usr/bin/env python3
"""P4 geometry-domain eval — pass@1 and the generalization gap on the new geometry
split (marc/eval/problems.py's geometry_in_distribution/geometry_held_out), per
results/p4_scale/scaling_notes.md's own "Next steps" item #2.

Writes results/p4_scale/geometry_eval.json and appends a summary section to
results/p4_scale/scaling_notes.md.

Usage:
    python scripts/run_geometry_eval.py
    python scripts/run_geometry_eval.py --n 25 --k 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.problems import geometry_held_out, geometry_in_distribution
from marc.eval.runner import run_split_eval
from marc.eval.solver import GradientRefinementSolver

# Tuned for the geometry domain's nonconvex quartic energy — noise off, smaller
# learning rate, much longer polish than the linear-system suites need. See
# results/p4_scale/roadmap.md for why (the default refine() hyperparameters,
# tuned against convex linear systems, solve ~0% of geometry instances).
GEOMETRY_REFINE_KWARGS = dict(
    steps=1200, lr=0.008, sigma0=0.0, noise=False,
    polish_steps=6000, polish_lr=0.02, init_scale=3.0,
)

NOTES_PATH = Path("results/p4_scale/scaling_notes.md")
OUT_PATH = Path("results/p4_scale/geometry_eval.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 geometry-domain eval")
    ap.add_argument("--n", type=int, default=25, help="problems per split")
    ap.add_argument("--k", type=int, default=12, help="candidates per problem")
    args = ap.parse_args()

    solver = GradientRefinementSolver(**GEOMETRY_REFINE_KWARGS)
    print(f"== P4 geometry eval :: solver=refine (geometry-tuned) n={args.n} k={args.k} ==")

    t0 = time.time()
    id_problems = geometry_in_distribution(n=args.n)
    ho_problems = geometry_held_out(n=args.n)
    metrics = run_split_eval(id_problems, ho_problems, solver=solver, n_samples=args.k, solver_name="refine")
    wall = time.time() - t0

    idm = metrics["splits"]["in_distribution"]
    hom = metrics["splits"]["held_out_structure"]
    print(f"[geometry_in_distribution] solve_rate={idm['solve_rate']:.2f}")
    print(f"[geometry_held_out]        solve_rate={hom['solve_rate']:.2f}")
    print(f"[generalization_gap]       {metrics['generalization_gap']:.3f}")
    print(f"wall={wall:.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({**metrics, "wall_seconds": round(wall, 2)}, indent=2))
    print(f"-> {OUT_PATH}")

    section = "\n".join([
        "",
        "## Geometry-domain eval (P4)",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "**Task:** `refine` baseline (geometry-tuned hyperparameters — see "
        "`scripts/run_geometry_eval.py`) on `marc/eval/problems.py`'s "
        "`geometry_in_distribution` (2-var triangle) / `geometry_held_out` "
        "(4-var, two-point chain) split.",
        "",
        "| Split | n | Solve rate |",
        "|---|---|---|",
        f"| geometry_in_distribution | {idm['n_problems']} | {idm['solve_rate']:.2f} |",
        f"| geometry_held_out | {hom['n_problems']} | {hom['solve_rate']:.2f} |",
        "",
        f"**Generalization gap:** {metrics['generalization_gap']:.3f}",
        "",
        "Unlike the linear-system suites (P1/P2), this domain's energy is a "
        "nonconvex quartic (squared-distance factors are quadratic in the "
        "unknowns), so the default `refine()` hyperparameters — tuned against "
        "convex linear systems — solve close to 0% of instances; noise off, a "
        "smaller learning rate, and a much longer polish (see "
        "`GEOMETRY_REFINE_KWARGS`) are needed to reach the checker's exact-rational "
        "tolerance. See `results/p4_scale/roadmap.md` for the full writeup.",
        "",
    ])
    with NOTES_PATH.open("a") as fh:
        fh.write(section)
    print(f"-> appended to {NOTES_PATH}")


if __name__ == "__main__":
    main()
