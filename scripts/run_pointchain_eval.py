"""Learned-vs-random eval on the coupled point-chain geometry family (R9 follow-up).

The crossover law (scripts/run_crossover_theory.py) measured a steep reachability
collapse on make_point_chain — a real-domain regime it flags as learning-favorable.
This script runs the actual arms on that family, for k in {1,2,3,4} points
(n = 2k variables):

  * langevin        : one Gaussian init, noisy refine, best-of-K
  * random_restart  : K Gaussian inits + deterministic polish, best-of-K (the
                      amortization control)
  * learned         : Stage-A checkpoint via LearnedSolver (DDIM, polish off),
                      proposals polished by the SAME refine, best-of-K; skipped
                      (still exit 0) when no checkpoint is available

All arms share the geometry-tuned polish from run_crossover_theory (the default
refine hyperparameters solve ~0% of geometry instances); the langevin arm differs
only in noise being on. Acceptance is the Checker gate on the candidate snapped to
a 6-decimal grid, exactly as the crossover measurement did. Every rate carries a
95% Wilson CI; learned-vs-random carries a two-proportion z p-value per k.

Outputs results/p_geometry/pointchain_eval.json.
Run:  python scripts/run_pointchain_eval.py [--quick] [--trials 40] [--K 8] [--ckpt PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

from marc.cas.checker import Checker
from marc.data.geometry import make_point_chain
from marc.eval.metrics import two_proportion_z, wilson_interval
from marc.eval.solver import load_solver
from marc.refine.iterative import refine

# Geometry-tuned polish, identical across arms (scripts/run_crossover_theory.py).
GEOMETRY_REFINE = dict(steps=1200, lr=0.008, sigma0=0.0, noise=False,
                       polish_steps=6000, polish_lr=0.02)
# The langevin arm turns exploration noise back on; everything else is shared.
LANGEVIN_REFINE = dict(GEOMETRY_REFINE, noise=True, sigma0=0.5)
INIT_SD = 3.0        # Gaussian init, matched to the geometry eval's init_scale
DECIMALS = 6         # snap-to-grid before the checker's exact gate
KS = [1, 2, 3, 4]    # points per chain; n = 2k variables

OUT_PATH = Path("results/p_geometry/pointchain_eval.json")


def accepted(chk: Checker, g, x) -> bool:
    return chk.verify(g, [round(v, DECIMALS) for v in x]).accepted


def _cell(k, n):
    return {"k": k, "n": n, "rate": k / n, "ci95": wilson_interval(k, n)}


def eval_k(k: int, trials: int, K: int, solver=None, seed0: int = 0):
    """Solved counts for one chain length. Returns {arm: (solved, trials)}."""
    if solver is not None:
        import torch
    chk = Checker()
    nv = 2 * k
    counts = {"langevin": 0, "random_restart": 0, "learned": 0}
    for j in range(trials):
        rng = random.Random(seed0 + 7919 * j)
        g, _sol = make_point_chain(k, rng)
        x0 = [rng.gauss(0, INIT_SD) for _ in range(nv)]
        counts["langevin"] += any(
            accepted(chk, g, refine(g, x0, seed=s + K * j, **LANGEVIN_REFINE).x)
            for s in range(K))
        solved = False
        for s in range(K):
            r = random.Random(9000 * s + 31 * j + k)
            xr = [r.gauss(0, INIT_SD) for _ in range(nv)]
            if accepted(chk, g, refine(g, xr, seed=0, **GEOMETRY_REFINE).x):
                solved = True
                break
        counts["random_restart"] += solved
        if solver is None:
            continue
        solved = False
        for s in range(K):
            torch.manual_seed(1000 * s + 31 * j + k)
            prop = solver.sample(SimpleNamespace(graph=g, metadata={}), 1)[0]
            if prop is None or not all(abs(v) < 1e6 for v in prop):
                continue  # rollout diverged; nothing to polish
            if accepted(chk, g, refine(g, prop, seed=0, **GEOMETRY_REFINE).x):
                solved = True
                break
        counts["learned"] += solved
    return {arm: (n_ok, trials) for arm, n_ok in counts.items()}


def run(ks, trials: int, K: int, ckpt: str | None = None) -> dict:
    # polish=False: the shared refine above stays the single polisher for every arm
    solver = load_solver("learned", checkpoint=ckpt, polish=False) if ckpt else None
    rows = []
    for k in ks:
        by_arm = eval_k(k, trials, K, solver=solver, seed0=100 + 101 * k)
        row = {"points": k, "n": 2 * k,
               "langevin": _cell(*by_arm["langevin"]),
               "random_restart": _cell(*by_arm["random_restart"])}
        if solver is not None:
            row["learned"] = _cell(*by_arm["learned"])
            _, row["p_learned_gt_random"] = two_proportion_z(
                row["learned"]["k"], row["learned"]["n"],
                row["random_restart"]["k"], row["random_restart"]["n"])
        else:
            row["learned"] = {"status": "skipped",
                              "reason": "no checkpoint: set --ckpt or MARC_CKPT"}
        print(f"  points={k} n={2 * k}: " + "  ".join(
            f"{arm}={row[arm]['k']}/{row[arm]['n']}" if "k" in row[arm] else f"{arm}=skipped"
            for arm in ("langevin", "random_restart", "learned")), flush=True)
        rows.append(row)
    return {"K": K, "trials": trials, "ckpt": ckpt,
            "learned_mode": "checkpoint" if ckpt else "skipped",
            "refine_kwargs": GEOMETRY_REFINE, "init_sd": INIT_SD, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Learned-vs-random eval on the point-chain geometry family (R9)")
    ap.add_argument("--trials", type=int, default=40, help="fresh instances per k")
    ap.add_argument("--K", type=int, default=8, help="best-of-K budget (all arms)")
    ap.add_argument("--quick", action="store_true", help="tiny run for CI (k=[1,2])")
    ap.add_argument("--ckpt", default=os.environ.get("MARC_CKPT"),
                    help="trained Stage-A checkpoint for the learned arm "
                         "(default $MARC_CKPT; arm skipped if unset)")
    args = ap.parse_args()

    ks = [1, 2] if args.quick else KS
    mode = f"ckpt {args.ckpt}" if args.ckpt else "learned arm skipped (no checkpoint)"
    print(f"Point-chain geometry eval — best-of-{args.K}, {args.trials} trials/k, "
          f"ks={ks}, {mode}")
    payload = run(ks, args.trials, args.K, ckpt=args.ckpt)

    print(f"\n{'pts':>4} {'n':>3} {'langevin':>9} {'random':>8} {'learned':>8} {'p(l>rand)':>10}")
    for r in payload["rows"]:
        learned = f"{r['learned']['rate']:.3f}" if "rate" in r["learned"] else "skipped"
        p = f"{r['p_learned_gt_random']:.4f}" if "p_learned_gt_random" in r else "-"
        print(f"{r['points']:>4} {r['n']:>3} {r['langevin']['rate']:>9.3f} "
              f"{r['random_restart']['rate']:>8.3f} {learned:>8} {p:>10}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
