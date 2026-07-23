#!/usr/bin/env python3
"""Is the probe's R28 edge selection information or stream diversity?

The probe solves 0.698 of two-stream test failures at ~19.5 restarts by
spending 1 restart per candidate, each on its own stream. Whether ANY
selector at the reference budget (top-1 construction + K_REF restarts) can
approach that depends on how the per-restart accept probability is spread
across the menu: concentrated on a few candidates -> selection has a real
ceiling; diffuse -> the probe's edge is the diversity of its independent
streams, and no selector at the reference budget can beat plain restarts.

Cross-fitted measurement on the cached v3 test split: estimate each
candidate's accept rate from 1 restart on each of two fresh streams
(salts +5/+6), select the argmax, then grade that pick at the reference
budget on a held-out stream (+7). Salts +0..+4 are spoken for by the
dataset/eval protocol; nothing here reuses them. The implied top-1 number
computed from the screening estimates themselves is reported only as the
optimistic (selection-on-noise) bound; the cross-fitted rate is the claim.

  python3 scripts/probe_concentration.py [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.metrics import rate_cell
from marc.structure.geo_repair import STREAM_SALT, make_dataset, solve_graph
from marc.structure.invention_data import REFERENCE_SOLVER

K_REF = REFERENCE_SOLVER["k_refine"]
SCREEN_SALTS = (5, 6)
GRADE_SALT = 7


def _one(inst):
    wins = []
    for j, c in enumerate(inst.constructions):
        g = c.apply(inst.graph)
        w = sum(solve_graph(g, seed=inst.seed + s * STREAM_SALT + 31 * j,
                            k_restarts=1) for s in SCREEN_SALTS)
        wins.append(w)
    pick = max(range(len(wins)), key=lambda j: (wins[j], -j))
    graded = solve_graph(inst.constructions[pick].apply(inst.graph),
                         seed=inst.seed + GRADE_SALT * STREAM_SALT,
                         k_restarts=K_REF)
    return {"id": inst.id, "k": inst.k, "n_candidates": len(wins),
            "wins": wins, "pick": inst.constructions[pick].name,
            "pick_wins": wins[pick], "graded_flip": bool(graded)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="R28 probe concentration control")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--out", default="results/p_geo_repair/probe_concentration.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    test = make_dataset(args.n_test, args.seed + 900000, ks=(10, 12, 14),
                        label_streams=3)
    print(f"test split: {len(test)} failures (cache)", flush=True)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_one, test, chunksize=4))
    else:
        rows = [_one(i) for i in test]

    def pool(rs, tag):
        n = len(rs)
        m = len(SCREEN_SALTS)
        p_max = [max(r["wins"]) / m for r in rs]
        implied = sum(1 - (1 - p) ** K_REF for p in p_max) / n
        return {
            "pool": tag, "n_failures": n,
            "crossfit_top1_at_kref": rate_cell(
                sum(r["graded_flip"] for r in rs), n),
            "implied_top1_at_kref_optimistic": implied,
            "mean_frac_candidates_with_any_win":
                sum(sum(w > 0 for w in r["wins"]) / r["n_candidates"]
                    for r in rs) / n,
            "mean_p_max": sum(p_max) / n,
            "frac_pick_zero_wins": sum(r["pick_wins"] == 0 for r in rs) / n,
        }

    out = {
        "status": "ok",
        "method": "cross-fitted selection ceiling at the reference budget: "
                  "2-stream 1-restart screen per candidate (salts +5/+6), "
                  "argmax pick graded at K_REF on held-out stream (+7)",
        "k_ref": K_REF, "screen_salts": SCREEN_SALTS, "grade_salt": GRADE_SALT,
        "config": vars(args),
        "pools": {"trained": pool([r for r in rows if r["k"] in (10, 12)], "trained"),
                  "transfer": pool([r for r in rows if r["k"] == 14], "transfer")},
        "rows": rows,
        "wall_s": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for tag, p in out["pools"].items():
        c = p["crossfit_top1_at_kref"]
        print(f"{tag}: crossfit top-1@{K_REF} = {c['rate']:.3f} "
              f"[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}] (N={p['n_failures']}), "
              f"implied-optimistic {p['implied_top1_at_kref_optimistic']:.3f}, "
              f"mean p_max {p['mean_p_max']:.3f}, "
              f"frac candidates with any win "
              f"{p['mean_frac_candidates_with_any_win']:.3f}, "
              f"picks with 0 wins {p['frac_pick_zero_wins']:.3f}", flush=True)
    print(f"wrote {args.out}; wall={out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
