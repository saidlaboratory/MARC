#!/usr/bin/env python3
"""Solve a single constraint graph and report the checker verdict.

Loads a FactorGraph JSON, runs a solver (default: the learned diffusion solver
with a trained checkpoint if one exists, else the gradient-refinement baseline),
and prints the best candidate plus the checker's accept/reject.

    python scripts/solve_one.py                         # two_equations, learned
    python scripts/solve_one.py --solver refine         # classical baseline
    python scripts/solve_one.py --sample path/to.json --k 16
"""
import argparse
from pathlib import Path

from marc.graph.serialize import load_graph
from marc.cas.checker import Checker
from marc.eval.runner import Problem
from marc.eval.solver import load_solver

# preference order for an auto-selected learned checkpoint
DEFAULT_CKPTS = [
    "checkpoints/denoiser_stage_b_standard.pt",
    "checkpoints/denoiser_stage_b_purist.pt",
    "checkpoints/denoiser_stage_a.pt",
]


def _pick_checkpoint(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for c in DEFAULT_CKPTS:
        if Path(c).exists():
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="MARC: solve one constraint graph and check it.")
    ap.add_argument("--sample", default="marc/data/examples/two_equations.json",
                    help="input FactorGraph JSON")
    ap.add_argument("--solver", default="refine", choices=["learned", "refine", "dummy"])
    ap.add_argument("--weights", default=None, help="denoiser checkpoint (learned solver)")
    ap.add_argument("--k", type=int, default=8, help="candidates / best-of-N rollouts")
    args = ap.parse_args()

    G = load_graph(args.sample)
    checker = Checker()
    # Checker verifies candidates against the graph's constraints; the placeholder
    # solution just fixes the expected dimensionality for the solver contract.
    problem = Problem(id=Path(args.sample).stem, graph=G, solution=[0.0] * len(G.variables))

    solver_name = args.solver
    kwargs: dict = {}
    if solver_name == "learned":
        ckpt = _pick_checkpoint(args.weights)
        if ckpt is None:
            print("No trained checkpoint found — falling back to --solver refine.")
            solver_name = "refine"
        else:
            print(f"Using learned solver with checkpoint: {ckpt}")
            kwargs["checkpoint"] = ckpt

    solver = load_solver(solver_name, **kwargs)
    print(f"Solving {args.sample} with '{solver_name}' (best-of-{args.k})…\n")
    candidates = solver.sample(problem, args.k)

    # first checker-accepted candidate wins; otherwise keep the lowest-residual one
    best, best_res, accepted = None, float("inf"), False
    for x in candidates:
        if x is None:
            continue
        res = checker.verify(G, x)
        if res.accepted:
            best, accepted = x, True
            break
        if res.max_residual < best_res:
            best, best_res = x, res.max_residual

    print("-" * 50)
    if best is None:
        print("Solver returned no finite candidate.")
        return
    names = [v.id for v in G.variables]
    print("Solution:", {n: round(float(v), 4) for n, v in zip(names, best)})
    if accepted:
        print("Checker: ACCEPTED ✅")
    else:
        print(f"Checker: REJECTED ❌  (max|residual| = {best_res:.3g})")


if __name__ == "__main__":
    main()
