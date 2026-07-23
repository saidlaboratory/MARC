#!/usr/bin/env python3
"""Oracle-marginal control on the geometry point-chain family (R29 for the second
coupled family).

R29 showed that on the coupled chained-bilinear family, sampling each coordinate
from its TRUE per-variable marginal still ties random restart — so marginals are
causally insufficient under coupling, and no better marginal learner could close
the gap. The geometry point chains (R25) are the paper's other coupled family:
reachability collapses, yet the trained denoiser only ties random. This runs the
same oracle control there. If oracle marginals also tie random, the law's causal
mechanism covers BOTH coupled families; if they win, the geometry tie was a
capacity failure of our denoiser, not coupling, and the paper must say so.

Protocol is R25's, reused from run_pointchain_learned: same test instances
(seed0=90000+k), same restart seeds (9000*s+31*j+k, CRN), same geometry polish +
6-decimal snap + Checker gate. The random arm is recomputed here and must
reproduce pointchain_learned.json digit-for-digit — a built-in cross-check. The
learned column is cited from that file (identical test set and budget).

Run:  PYTHONPATH=. python3 scripts/run_oracle_marginal_pointchain.py [--trials 40 --K 8]
Writes results/p_geometry/oracle_marginal_pointchain.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from run_pointchain_learned import suite, accepted, POLISH, INIT_SD

from marc.cas.checker import Checker
from marc.eval.metrics import rate_cell, two_proportion_z
from marc.refine.iterative import refine

KS = [1, 2, 3, 4]                 # points per chain; n = 2k
PC_JSON = Path("results/p_geometry/pointchain_learned.json")


def marginal_pools(k, count, pool_seed0):
    items = suite(k, count, pool_seed0)          # disjoint from the 90000+k test range
    return [[sol[i] for _, sol in items] for i in range(2 * k)]


def count_arm(k, test, K, draw):
    chk = Checker(); ok = 0
    for j, (g, _sol) in enumerate(test):
        nv = 2 * k
        for s in range(K):
            r = random.Random(9000 * s + 31 * j + k)
            x0 = draw(r, nv)
            if accepted(chk, g, refine(g, x0, seed=0, **POLISH).x):
                ok += 1
                break
    return ok, len(test)


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-marginal control, geometry point chains")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--pool", type=int, default=200)
    args = ap.parse_args()

    pc = {r["points"]: r for r in json.loads(PC_JSON.read_text())["rows"]}
    t0 = time.time()
    print(f"Oracle-marginal vs random vs learned(R25) — best-of-{args.K}, {args.trials} test/k")
    print(f"{'k':>3} {'random':>8} {'oracle':>8} {'learned':>8} {'p(o>rand)':>10} {'repro':>6}")
    rows = []
    for k in KS:
        test = suite(k, args.trials, seed0=90000 + k)
        pools = marginal_pools(k, args.pool, pool_seed0=100 + k)
        cr = count_arm(k, test, args.K, lambda r, nv: [r.gauss(0, INIT_SD) for _ in range(nv)])
        co = count_arm(k, test, args.K, lambda r, nv: [r.choice(pools[i]) for i in range(nv)])
        row = pc[k]
        lk, ln = row["learned"]["k"], row["learned"]["n"]
        repro = abs(cr[0] / cr[1] - row["random_restart"]["rate"]) < 1e-12
        _, p_rand = two_proportion_z(co[0], co[1], cr[0], cr[1])
        rows.append({"points": k, "n": 2 * k,
                     "random": rate_cell(*cr), "oracle_marginal": rate_cell(*co),
                     "learned_r25": {"k": lk, "n": ln, "rate": row["learned"]["rate"]},
                     "p_oracle_gt_random": p_rand, "random_reproduces_r25": repro})
        print(f"{k:>3} {cr[0]/cr[1]:>8.3f} {co[0]/co[1]:>8.3f} {lk/ln:>8.3f} "
              f"{p_rand:>10.4f} {str(repro):>6}", flush=True)

    n_win = sum(r["p_oracle_gt_random"] < 0.05
                and r["oracle_marginal"]["rate"] > r["random"]["rate"] for r in rows)
    verdict = ("oracle marginals BEAT random on geometry — the tie was a denoiser capacity "
               "failure, not coupling; soften the law's geometry claim" if n_win >= 2 else
               "oracle marginals tie random on geometry too — marginals are causally "
               "insufficient under coupling on BOTH families; the law's mechanism holds")
    print(f"\noracle significantly beats random on {n_win}/{len(rows)} chain lengths → {verdict}")
    out = Path("results/p_geometry/oracle_marginal_pointchain.json")
    out.write_text(json.dumps({
        "status": "ok",
        "method": "oracle-marginal proposal control on the geometry point-chain family "
                  "(each coordinate drawn from the true per-variable marginal, pooled from "
                  "disjoint training-side chains; identical best-of-K geometry polish + "
                  "6-decimal snap + Checker gate and restart seeds as the random arm)",
        "config": {**vars(args), "ks": KS, "learned_source": str(PC_JSON)},
        "rows": rows, "verdict": verdict, "wall_s": time.time() - t0}, indent=2))
    print(f"wrote {out}; wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
