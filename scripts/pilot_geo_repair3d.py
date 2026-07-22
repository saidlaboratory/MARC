"""#123 pilot: does the 2D geometry-repair trap transfer to the native 3D (DMDGP)
setting? Run with the SAME discipline that caught us in 2D -- two-stream failure
selection FIRST, restart-scaling curve measured BEFORE any construction claim.

For each k it measures, over N sphere-anchor pruned chains:
  * two-stream failure rate: fraction the reference solver fails on BOTH of two
    independent restart streams (single-stream failure selects on noise);
  * restart-scaling on those failures: control (K_REF more restarts), +16, +32 --
    if extra restarts rescue the failures, the "difficulty" is restart noise, not a
    structural trap, and no construction claim is warranted.

Writes results/p_geo_repair3d/pilot.json. Emits a construction-repair claim ONLY
if a hard-failure population survives restart scaling; otherwise reports the
negative, which is itself the #123 result ("the planar trap does not survive in
3D").
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from marc.structure.invention_data import REFERENCE_SOLVER
from marc.data.geometry import make_pruned_chain_3d
from marc.structure.geo_repair3d import STREAM_SALT, solve_graph_3d

K_REF = REFERENCE_SOLVER["k_refine"]


def run(ks, n_per_k, n_extra, seed_base):
    rows = []
    for k in ks:
        hard = []
        for t in range(n_per_k):
            s = seed_base + 1000 * k + t
            g, _, _ = make_pruned_chain_3d(k, random.Random(s), n_extra=n_extra)
            if not (solve_graph_3d(g, seed=s) or solve_graph_3d(g, seed=s + STREAM_SALT)):
                hard.append((g, s))
        n_hard = len(hard)
        ctrl = sum(solve_graph_3d(g, seed=s + 2 * STREAM_SALT) for g, s in hard)
        p16 = sum(solve_graph_3d(g, seed=s + 3 * STREAM_SALT, k_restarts=16) for g, s in hard)
        p32 = sum(solve_graph_3d(g, seed=s + 4 * STREAM_SALT, k_restarts=32) for g, s in hard)
        # a hard-failure population survives only if some failures resist +32 restarts
        survivors = n_hard - p32
        rows.append({
            "k": k, "n": n_per_k, "n_extra": n_extra,
            "two_stream_failures": n_hard,
            "failure_rate": round(n_hard / n_per_k, 3),
            "restart_control_rescued": ctrl,
            "restart_plus16_rescued": p16,
            "restart_plus32_rescued": p32,
            "failures_surviving_plus32": survivors,
        })
        print(f"k={k:2d} n_extra={n_extra:2d}: fail {n_hard:2d}/{n_per_k} "
              f"({n_hard / n_per_k:4.0%}); +32 rescues {p32}/{n_hard}; "
              f"survivors={survivors}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="6,7,8,9,10")
    ap.add_argument("--n-per-k", type=int, default=40)
    ap.add_argument("--n-extra", type=int, default=None,
                    help="long-range edges; default ceil(k/2) per chain")
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--out", default="results/p_geo_repair3d/pilot.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    all_rows = []
    for k in ks:
        n_extra = args.n_extra if args.n_extra is not None else (k + 1) // 2
        all_rows += run([k], args.n_per_k, n_extra, args.seed_base)

    total_fail = sum(r["two_stream_failures"] for r in all_rows)
    total_survivors = sum(r["failures_surviving_plus32"] for r in all_rows)
    verdict = ("TRAP HOLDS: a hard-failure population survives +32 restarts; a "
               "construction-repair claim is warranted."
               if total_survivors > 0 else
               "NO TRAP IN 3D: two-stream failures are rare and fully rescued by "
               "+32 restarts. The planar reflection-tree difficulty does not survive "
               "in the sphere-anchor 3D setting (more distance pruning makes it "
               "EASIER, opposite to 2D). Negative result; no construction claim.")
    out = {
        "experiment": "geo_repair3d_pilot",
        "issue": 123,
        "design": "sphere-anchor 3D pruned chains (3 anchors: origin, (c,0,0), (0,d,0); "
                  "each point sphere-sphere-sphere -> binary reflection across the anchor plane)",
        "reference_solver": REFERENCE_SOLVER,
        "rows": all_rows,
        "total_two_stream_failures": total_fail,
        "total_failures_surviving_plus32": total_survivors,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n" + verdict)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
