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
from marc.data.geometry import make_clique_chain_3d, make_pruned_chain_3d
from marc.structure.geo_repair3d import STREAM_SALT, solve_graph_3d

K_REF = REFERENCE_SOLVER["k_refine"]

BUILDERS = {"anchor": make_pruned_chain_3d, "clique": make_clique_chain_3d}


#: streams used to confirm a survivor is genuinely hard, not restart-noise. A
#: single +32 stream is NOT enough: the two-stream selection already showed that
#: ~half of one-stream failures solve on a fresh stream, so a "survivor" of one
#: +32 stream must be re-confirmed on independent streams before it counts.
CONFIRM_STREAMS = (4, 6, 8)


def run(ks, n_per_k, n_extra, seed_base, design="anchor"):
    build = BUILDERS[design]
    rows = []
    for k in ks:
        hard = []
        for t in range(n_per_k):
            s = seed_base + 1000 * k + t
            g, _, _ = build(k, random.Random(s), n_extra=n_extra)
            if not (solve_graph_3d(g, seed=s) or solve_graph_3d(g, seed=s + STREAM_SALT)):
                hard.append((g, s))
        n_hard = len(hard)
        ctrl = sum(solve_graph_3d(g, seed=s + 2 * STREAM_SALT) for g, s in hard)
        p16 = sum(solve_graph_3d(g, seed=s + 3 * STREAM_SALT, k_restarts=16) for g, s in hard)
        # a GENUINE hard instance survives +32 restarts on EVERY confirmation stream;
        # anything a single fresh +32 stream rescues was restart-noise, not a trap.
        genuine = sum(
            not any(solve_graph_3d(g, seed=s + m * STREAM_SALT, k_restarts=32)
                    for m in CONFIRM_STREAMS)
            for g, s in hard
        )
        rows.append({
            "k": k, "n": n_per_k, "n_extra": n_extra,
            "two_stream_failures": n_hard,
            "failure_rate": round(n_hard / n_per_k, 3),
            "restart_control_rescued": ctrl,
            "restart_plus16_rescued": p16,
            "genuine_hard_survive_plus32_all_streams": genuine,
            "genuine_hard_rate": round(genuine / n_per_k, 3),
        })
        print(f"k={k:2d} n_extra={n_extra:2d}: two-stream-fail {n_hard:2d}/{n_per_k} "
              f"({n_hard / n_per_k:4.0%}); GENUINE hard (survive +32 x{len(CONFIRM_STREAMS)}) "
              f"{genuine}/{n_per_k} ({genuine / n_per_k:.0%})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=list(BUILDERS), default="anchor",
                    help="anchor = sphere-anchor chain; clique = consecutive-clique DMDGP")
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
        all_rows += run([k], args.n_per_k, n_extra, args.seed_base, design=args.design)

    total_fail = sum(r["two_stream_failures"] for r in all_rows)
    total_genuine = sum(r["genuine_hard_survive_plus32_all_streams"] for r in all_rows)
    total_n = sum(r["n"] for r in all_rows)
    # a construction-repair claim needs a genuine hard POPULATION, not a stray
    # instance: require the confirmed-hard rate to clear a real threshold.
    genuine_rate = total_genuine / total_n if total_n else 0.0
    trap = genuine_rate >= 0.10
    verdict = ("TRAP HOLDS: a genuine hard-failure population survives +32 restarts "
               f"across independent streams ({genuine_rate:.0%}); a construction-repair "
               "claim is warranted."
               if trap else
               f"NO TRAP IN 3D ({args.design}): two-stream failures occur but are restart-"
               f"noise -- only {total_genuine}/{total_n} ({genuine_rate:.1%}) survive +32 "
               "restarts across independent streams, no population. The planar "
               "reflection-tree difficulty does not survive in 3D; a bit more restart "
               "budget always rescues it. Negative result; no construction claim.")
    out = {
        "experiment": "geo_repair3d_pilot",
        "issue": 123,
        "design": args.design,
        "design_note": ("sphere-anchor: 3 fixed anchors, each point sphere^3 -> reflection "
                        "across anchor plane" if args.design == "anchor" else
                        "consecutive-clique DMDGP: each point i>=3 pinned to its 3 "
                        "predecessors -> propagating 2^(k-3) reflection tree"),
        "reference_solver": REFERENCE_SOLVER,
        "rows": all_rows,
        "total_two_stream_failures": total_fail,
        "total_genuine_hard": total_genuine,
        "genuine_hard_rate": round(genuine_rate, 4),
        "confirm_streams": len(CONFIRM_STREAMS),
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n" + verdict)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
