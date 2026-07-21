#!/usr/bin/env python3
"""P4 geometry-domain eval — pass@1 and the generalization gap on the new geometry
split (marc/eval/problems.py's geometry_in_distribution/geometry_held_out), per
results/p4_scale/scaling_notes.md's own "Next steps" item #2.

Three arms (same protocol as scripts/run_hard_eval.py, best-of-K, exact accept):

  * refine  — the geometry-tuned GradientRefinementSolver baseline (unchanged)
  * random  — K uniform random inits + the same deterministic polish, no learning
  * learned — LearnedSolver (diffusion proposal + polish) from a checkpoint;
              skipped (still exit 0) when no checkpoint is available

Writes results/p4_scale/geometry_eval.json and appends a summary section to
results/p4_scale/scaling_notes.md.

Usage:
    python scripts/run_geometry_eval.py
    python scripts/run_geometry_eval.py --n 25 --k 12
    MARC_CKPT=checkpoints/denoiser.pt python scripts/run_geometry_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.cas.checker import Checker
from marc.eval.metrics import two_proportion_z, wilson_interval
from marc.eval.problems import geometry_held_out, geometry_in_distribution
from marc.eval.runner import run_split_eval
from marc.eval.solver import GradientRefinementSolver, load_solver
from marc.refine.iterative import refine

# Tuned for the geometry domain's nonconvex quartic energy — noise off, smaller
# learning rate, much longer polish than the linear-system suites need. See
# results/p4_scale/roadmap.md for why (the default refine() hyperparameters,
# tuned against convex linear systems, solve ~0% of geometry instances).
GEOMETRY_REFINE_KWARGS = dict(
    steps=1200, lr=0.008, sigma0=0.0, noise=False,
    polish_steps=6000, polish_lr=0.02, init_scale=3.0,
)
# the same deterministic descent, minus the solver-level init knob
_POLISH_KWARGS = {k: v for k, v in GEOMETRY_REFINE_KWARGS.items() if k != "init_scale"}

NOTES_PATH = Path("results/p4_scale/scaling_notes.md")
OUT_PATH = Path("results/p4_scale/geometry_eval.json")


def _cell(k, n):
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def _arm(id_kn, ho_kn):
    return {
        "in_distribution": _cell(*id_kn),
        "held_out_structure": _cell(*ho_kn),
        "pooled": _cell(id_kn[0] + ho_kn[0], id_kn[1] + ho_kn[1]),
    }


def _refine_counts(per_problem, split):
    rows = [pp for pp in per_problem if pp["split"] == split]
    return sum(pp["first_success_index"] is not None for pp in rows), len(rows)


def random_arm(problems, K, scale=GEOMETRY_REFINE_KWARGS["init_scale"]):
    """Control: K uniform random inits + deterministic polish, best-of-K, no
    learning — run_hard_eval.random_count with the geometry-tuned descent."""
    chk = Checker()
    ok = 0
    for i, p in enumerate(problems):
        nv = len(p.graph.variables)
        solved = False
        for s in range(K):
            r = random.Random(7000 * s + i)
            x0 = [r.uniform(-scale, scale) for _ in range(nv)]
            if chk.verify(p.graph, refine(p.graph, x0, seed=0, **_POLISH_KWARGS).x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok, len(problems)


def learned_arm(problems, solver, K):
    """K independent diffusion rollouts (LearnedSolver polishes internally),
    best-of-K. sample() may yield None when every rollout diverged — skipped."""
    chk = Checker()
    ok = 0
    for p in problems:
        solved = False
        for _ in range(K):
            x = solver.sample(p, 1)[0]
            if x is not None and chk.verify(p.graph, x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok, len(problems)


def build_payload(id_problems, ho_problems, k, refine_solver=None,
                  learned_solver=None, ckpt=None):
    solver = refine_solver or GradientRefinementSolver(**GEOMETRY_REFINE_KWARGS)
    metrics = run_split_eval(id_problems, ho_problems, solver=solver,
                             n_samples=k, solver_name="refine")
    pp = metrics["per_problem"]
    n_all = len(id_problems) + len(ho_problems)
    # refine's per-problem timing already comes from the runner; sum both splits
    refine_ms = (metrics["splits"]["in_distribution"]["wall_ms_total"]
                 + metrics["splits"]["held_out_structure"]["wall_ms_total"])
    t0 = time.perf_counter()
    rand_id, rand_ho = random_arm(id_problems, k), random_arm(ho_problems, k)
    rand_ms = (time.perf_counter() - t0) * 1000.0
    arms = {
        "refine": {**_arm(_refine_counts(pp, "geometry_in_distribution"),
                          _refine_counts(pp, "geometry_held_out")),
                   "wall_ms_total": refine_ms, "wall_ms_mean": refine_ms / n_all},
        "random": {**_arm(rand_id, rand_ho),
                   "wall_ms_total": rand_ms, "wall_ms_mean": rand_ms / n_all},
    }
    learned_vs_random = None
    if learned_solver is not None:
        t0 = time.perf_counter()
        l_id = learned_arm(id_problems, learned_solver, k)
        l_ho = learned_arm(ho_problems, learned_solver, k)
        learned_ms = (time.perf_counter() - t0) * 1000.0
        arms["learned"] = {"status": "ok", "checkpoint": ckpt,
                           **_arm(l_id, l_ho),
                           "wall_ms_total": learned_ms,
                           "wall_ms_mean": learned_ms / n_all}
        z, p = two_proportion_z(
            arms["learned"]["pooled"]["k"], arms["learned"]["pooled"]["n"],
            arms["random"]["pooled"]["k"], arms["random"]["pooled"]["n"])
        learned_vs_random = {"z": z, "p_one_sided": p}
    else:
        arms["learned"] = {"status": "skipped",
                           "reason": "no checkpoint: set --ckpt or MARC_CKPT"}
    return {**metrics, "arms": arms, "learned_vs_random": learned_vs_random}


def _arm_row(name, arm):
    if arm.get("status") == "skipped":
        return f"| {name} | — | skipped ({arm['reason']}) |"
    c = arm["pooled"]
    return (f"| {name} | {c['k']}/{c['n']} | "
            f"{c['rate']:.2f} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] |")


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 geometry-domain eval")
    ap.add_argument("--n", type=int, default=25, help="problems per split")
    ap.add_argument("--k", type=int, default=12, help="candidates per problem")
    ap.add_argument("--ckpt", default=os.environ.get("MARC_CKPT"),
                    help="denoiser checkpoint for the learned arm "
                         "(default: $MARC_CKPT; arm skipped if unset)")
    args = ap.parse_args()

    learned_solver = None
    if args.ckpt:
        import torch
        torch.manual_seed(0)
        learned_solver = load_solver("learned", checkpoint=args.ckpt)

    print(f"== P4 geometry eval :: arms=refine/random"
          f"{'/learned' if learned_solver else ''} n={args.n} k={args.k} ==")

    t0 = time.time()
    id_problems = geometry_in_distribution(n=args.n)
    ho_problems = geometry_held_out(n=args.n)
    payload = build_payload(id_problems, ho_problems, args.k,
                            learned_solver=learned_solver, ckpt=args.ckpt)
    wall = time.time() - t0

    idm = payload["splits"]["in_distribution"]
    hom = payload["splits"]["held_out_structure"]
    print(f"[geometry_in_distribution] solve_rate={idm['solve_rate']:.2f}")
    print(f"[geometry_held_out]        solve_rate={hom['solve_rate']:.2f}")
    print(f"[generalization_gap]       {payload['generalization_gap']:.3f}")
    for name, arm in payload["arms"].items():
        if arm.get("status") == "skipped":
            print(f"[arm {name}] skipped — {arm['reason']}")
            continue
        c = arm["pooled"]
        print(f"[arm {name}] best-of-{args.k} {c['k']}/{c['n']} = {c['rate']:.2f} "
              f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] {arm['wall_ms_mean']:.0f} ms/prob")
    if payload["learned_vs_random"]:
        lv = payload["learned_vs_random"]
        print(f"[learned vs random] z={lv['z']:.2f} p={lv['p_one_sided']:.3f}")
    print(f"wall={wall:.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({**payload, "wall_seconds": round(wall, 2)}, indent=2))
    print(f"-> {OUT_PATH}")

    section = [
        "",
        "## Geometry-domain eval (P4)",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "**Task:** `refine` baseline (geometry-tuned hyperparameters — see "
        "`scripts/run_geometry_eval.py`) on `marc/eval/problems.py`'s "
        "`geometry_in_distribution` (2-var triangle) / `geometry_held_out` "
        "(4-var, two-point chain) split, plus a random-restart control and a "
        "learned (diffusion + polish) arm.",
        "",
        "| Split | n | Solve rate |",
        "|---|---|---|",
        f"| geometry_in_distribution | {idm['n_problems']} | {idm['solve_rate']:.2f} |",
        f"| geometry_held_out | {hom['n_problems']} | {hom['solve_rate']:.2f} |",
        "",
        f"**Generalization gap:** {payload['generalization_gap']:.3f}",
        "",
        f"| Arm (best-of-{args.k}, pooled) | k/n | Rate [95% CI] |",
        "|---|---|---|",
        *(_arm_row(name, arm) for name, arm in payload["arms"].items()),
        "",
    ]
    if payload["learned_vs_random"]:
        lv = payload["learned_vs_random"]
        section += [f"**learned vs random (one-sided z):** z={lv['z']:.2f}, "
                    f"p={lv['p_one_sided']:.3f}", ""]
    section += [
        "Unlike the linear-system suites (P1/P2), this domain's energy is a "
        "nonconvex quartic (squared-distance factors are quadratic in the "
        "unknowns), so the default `refine()` hyperparameters — tuned against "
        "convex linear systems — solve close to 0% of instances; noise off, a "
        "smaller learning rate, and a much longer polish (see "
        "`GEOMETRY_REFINE_KWARGS`) are needed to reach the checker's exact-rational "
        "tolerance. See `results/p4_scale/roadmap.md` for the full writeup.",
        "",
    ]
    with NOTES_PATH.open("a") as fh:
        fh.write("\n".join(section))
    print(f"-> appended to {NOTES_PATH}")


if __name__ == "__main__":
    main()
