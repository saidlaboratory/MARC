"""Oracle-marginal control on the geometry point-chain family (#125 part 3).

The R9 factorization law explains the coupled null by saying a learned value proposal
can only amortize per-variable marginals, which carry no information about a joint
solution. run_oracle_marginal.py made the decisive test on the coupled
chained-bilinear family; this script runs the SAME logic on the geometry point-chain
family (make_point_chain), the other coupled family the law covers. If oracle
marginals ALSO tie random restart here, the law's causal control covers BOTH coupled
families, architecture-free.

Hand the proposal the family's TRUE per-variable marginals -- pool the solution
coordinate values of training-side instances at the same k (the same slice the learned
model trained on, seed-disjoint from test) and sample each coordinate independently --
then run the identical best-of-K geometry polish + Checker gate as the random arm. That
arm is a ceiling for any marginal learner: if it still ties random, no better marginal
learner could close the gap.

The DECISIVE comparison is internally valid by construction: oracle and random are
computed together in THIS run under common random numbers (same test instances, same
restart seeds 9000*s+31*j+k, same geometry polish + Checker), so oracle-vs-random does
not depend on any external file. As a bonus cross-check the random arm is recomputed
with run_pointchain_learned.py's exact seeds and compared to that file's random_restart;
a mismatch only means the committed pointchain_learned.json predates the current polish
presets (it does not affect the oracle-vs-random claim). The learned column is cited
from that file for context.

Run:  PYTHONPATH=. python3 scripts/run_oracle_marginal_geometry.py [--quick]
Writes results/p_geometry/oracle_marginal_pointchain.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from marc.cas.checker import Checker
from marc.data.geometry import make_point_chain
from marc.eval.metrics import rate_cell, two_proportion_z
from marc.refine.iterative import refine
from marc.refine.presets import GEOMETRY_INIT_SD as INIT_SD
from marc.refine.presets import GEOMETRY_POLISH_KWARGS as POLISH

LEARNED_JSON = Path("results/p_geometry/pointchain_learned.json")


def suite(k, count, seed0):
    # identical instance stream to run_pointchain_learned.suite
    return [make_point_chain(k, random.Random(seed0 + 7919 * j)) for j in range(count)]


def accepted(chk, g, x):
    return chk.verify(g, x).accepted


def random_count(test, k, K):
    """Reproduces run_pointchain_learned's random_restart arm (same seeds/polish)."""
    chk = Checker(); ok = 0
    for j, (g, _) in enumerate(test):
        nv = 2 * k
        for s in range(K):
            r = random.Random(9000 * s + 31 * j + k)
            xr = [r.gauss(0, INIT_SD) for _ in range(nv)]
            if accepted(chk, g, refine(g, xr, seed=0, **POLISH).x):
                ok += 1; break
    return ok, len(test)


def marginal_pools(k, pool_count, seed0):
    """Per-coordinate pool of TRUE solution values from training-side instances."""
    items = suite(k, pool_count, seed0)
    nv = 2 * k
    return [[sol[i] for _, sol in items] for i in range(nv)]


def oracle_count(test, pools, k, K):
    """random_count with each coordinate drawn from its true marginal instead of
    gauss(0, INIT_SD); identical restart seeds, polish, and Checker gate."""
    chk = Checker(); ok = 0
    for j, (g, _) in enumerate(test):
        nv = 2 * k
        for s in range(K):
            r = random.Random(9000 * s + 31 * j + k)
            x0 = [r.choice(pools[i]) for i in range(nv)]
            if accepted(chk, g, refine(g, x0, seed=0, **POLISH).x):
                ok += 1; break
    return ok, len(test)


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-marginal control on the geometry point-chain family")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=None, help="default: K from pointchain_learned.json")
    ap.add_argument("--pool", type=int, default=None, help="default: ntrain from pointchain_learned.json")
    args = ap.parse_args()

    learned = json.loads(LEARNED_JSON.read_text())
    K = args.K or learned["K"]
    pool_count = args.pool or learned.get("ntrain", 200)
    rows_by_k = {r["points"]: r for r in learned["rows"]}
    ks = sorted(rows_by_k)[:2] if args.quick else sorted(rows_by_k)
    trials = learned["rows"][0]["random_restart"]["n"]

    t0 = time.time()
    print(f"Oracle-marginal vs random vs learned(pointchain) -- best-of-{K}, {trials} test/k, "
          f"pool {pool_count}/k")
    print(f"{'k':>3} {'random':>8} {'oracle':>8} {'learned':>8} {'p(o>rand)':>10} {'p(o>lrn)':>9} {'rand==json'}")
    rows = []
    for k in ks:
        # primary rep (rep=0, off=0), matching run_pointchain_learned's cited rows
        test = suite(k, trials, seed0=90000 + k)
        cr = random_count(test, k, K)
        pools = marginal_pools(k, pool_count, seed0=100 + k)  # learned model's own training slice
        co = oracle_count(test, pools, k, K)
        lk, ln = rows_by_k[k]["learned"]["k"], rows_by_k[k]["learned"]["n"]
        _, p_rand = two_proportion_z(co[0], co[1], cr[0], cr[1])
        _, p_lrn = two_proportion_z(co[0], co[1], lk, ln)
        match = abs(cr[0] / cr[1] - rows_by_k[k]["random_restart"]["rate"]) < 1e-9
        rows.append({"k": k, "n_points": k, "n_vars": 2 * k,
                     "random": rate_cell(*cr),
                     "oracle_marginal": rate_cell(*co),
                     "learned_pointchain": {"k": lk, "n": ln,
                                            "rate": rows_by_k[k]["learned"]["rate"],
                                            "ci95": rows_by_k[k]["learned"]["ci95"]},
                     "p_oracle_gt_random": p_rand,
                     "p_oracle_gt_learned": p_lrn,
                     "random_reproduces_pointchain": match})
        print(f"{k:>3} {cr[0]/cr[1]:>8.3f} {co[0]/co[1]:>8.3f} {lk/ln:>8.3f} "
              f"{p_rand:>10.4f} {p_lrn:>9.4f}   {match}", flush=True)

    n_win = sum(r["p_oracle_gt_random"] < 0.05
                and r["oracle_marginal"]["rate"] > r["random"]["rate"] for r in rows)
    all_match = all(r["random_reproduces_pointchain"] for r in rows)
    verdict = ("oracle marginals BEAT random on point chains -- the law's mechanism story "
               "does not cover this coupled family; soften"
               if n_win >= 2 else
               "oracle marginals tie random on point chains -- marginals are causally "
               "insufficient under coupling HERE too; the law's causal control covers BOTH "
               "coupled families (coupled chained-bilinear AND geometry point chains)")
    print(f"\noracle significantly beats random on {n_win}/{len(rows)} k -> {verdict}")
    print(f"[cross-check] random arm reproduces committed pointchain_learned.json on all k: "
          f"{all_match}" + ("" if all_match else
          " -- committed json predates current polish presets (regenerate it); "
          "does NOT affect the oracle-vs-random CRN claim above"))

    payload = {
        "status": "ok",
        "issue": 125,
        "method": "oracle-marginal proposal control on the geometry point-chain family "
                  "(each coordinate drawn independently from the true per-variable marginal, "
                  "pooled from the learned model's own training slice; identical best-of-K "
                  "geometry polish + Checker gate and restart seeds as the random arm)",
        "config": {"K": K, "pool": pool_count, "trials": trials, "ks": ks,
                   "learned_source": str(LEARNED_JSON)},
        "seed_hygiene": {"pool_seed0": "100+k", "test_seed0": "90000+k",
                         "note": "pool = learned model's training slice (100+k), test = 90000+k; "
                                 "instance streams disjoint; restart seeds 9000*s+31*j+k shared "
                                 "with the random arm (CRN)"},
        "rows": rows,
        "random_reproduces_pointchain_all_k": all_match,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    out = Path("results/p_geometry/oracle_marginal_pointchain.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}; wall={payload['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
