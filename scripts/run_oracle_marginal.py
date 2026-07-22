"""Oracle-marginal control on the coupled chained-bilinear family.

The factorization law (R9) explains the R7 coupled null by saying a learned value
proposal can only amortize per-variable marginals, which carry no information about
a joint solution. Cheapest decisive test: hand the proposal the family's TRUE
per-variable marginals — pool the solution values of training-side instances at the
same n (seed-disjoint from test) and sample each coordinate independently — then run
the identical best-of-8 polish + checker. That arm is a ceiling for *any* marginal
learner. If oracle marginals still tie random restart, the mechanism is causal (no
better marginal learner could close the gap); if they win, the law's mechanism story
is wrong and the paper must soften.

Protocol is R7's, reused from run_coupled_eval: same test instances (seed0=500000,
60/n), same restart seeds (9000*s+nv, CRN across arms), same refine(noise=False) +
Checker gate. The random arm is fully deterministic, so its recomputation here must
reproduce results/p_coupled/coupled.json digit-for-digit — a built-in cross-check.
The learned column is cited from that same file (identical test set and budget), so
no training happens here.

Run:  python scripts/run_oracle_marginal.py [--quick]
Writes results/p_scaling/oracle_marginal.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from run_coupled_eval import gen, random_count

from marc.cas.checker import Checker
from marc.eval.metrics import rate_cell, two_proportion_z
from marc.refine.iterative import refine

POOL_SEED0 = 0        # first slice of the learned arm's own training range
TEST_SEED0 = 500000   # run_coupled_eval test range
COUPLED_JSON = Path("results/p_coupled/coupled.json")


def marginal_pools(n, count):
    items = gen(n, count, seed0=POOL_SEED0)
    return [[sol[i] for _, sol in items] for i in range(n)]


def oracle_count(items, pools, K):
    # random_count with r.uniform(-4,4) swapped for a draw from the true marginal;
    # same restart seeds, same polish, same checker.
    chk = Checker(); ok = 0
    for g, sol in items:
        nv = len(sol); solved = False
        for s in range(K):
            r = random.Random(9000 * s + nv)
            x0 = [r.choice(pools[i]) for i in range(nv)]
            if chk.verify(g, refine(g, x0, noise=False).x).accepted:
                solved = True; break
        ok += int(solved)
    return ok, len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-marginal control on the coupled family")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--pool", type=int, default=200)
    args = ap.parse_args()
    ns = [2, 3] if args.quick else [2, 3, 4, 6, 8]

    r7 = {r["n"]: r for r in json.loads(COUPLED_JSON.read_text())["rows"]}
    overlap = len(set(range(POOL_SEED0, POOL_SEED0 + args.pool))
                  & set(range(TEST_SEED0, TEST_SEED0 + args.test)))
    assert overlap == 0, "marginal pool overlaps test instances"

    t0 = time.time()
    print(f"Oracle-marginal vs random vs learned(R7) — best-of-{args.K}, {args.test} test/n, "
          f"pool {args.pool}/n")
    print(f"{'n':>3} {'random':>8} {'oracle':>8} {'learned':>8} {'p(o>rand)':>10} {'p(o>lrn)':>9}")
    rows = []
    for n in ns:
        test = gen(n, args.test, seed0=TEST_SEED0)
        cr = random_count(test, args.K)
        co = oracle_count(test, marginal_pools(n, args.pool), args.K)
        lk, ln = r7[n]["learned"]["k"], r7[n]["learned"]["n"]
        _, p_rand = two_proportion_z(co[0], co[1], cr[0], cr[1])
        _, p_lrn = two_proportion_z(co[0], co[1], lk, ln)
        match = abs(cr[0] / cr[1] - r7[n]["random"]["rate"]) < 1e-12
        if not match:
            print(f"  WARNING: n={n} random arm does not reproduce coupled.json "
                  f"({cr[0]/cr[1]:.3f} vs {r7[n]['random']['rate']:.3f})")
        rows.append({"n": n,
                     "random": rate_cell(*cr),
                     "oracle_marginal": rate_cell(*co),
                     "learned_r7": {"k": lk, "n": ln, "rate": r7[n]["learned"]["rate"],
                                    "ci95": r7[n]["learned"]["ci95"]},
                     "p_oracle_gt_random": p_rand,
                     "p_oracle_gt_learned": p_lrn,
                     "random_reproduces_r7": match})
        print(f"{n:>3} {cr[0]/cr[1]:>8.3f} {co[0]/co[1]:>8.3f} {lk/ln:>8.3f} "
              f"{p_rand:>10.4f} {p_lrn:>9.4f}", flush=True)

    n_win = sum(r["p_oracle_gt_random"] < 0.05
                and r["oracle_marginal"]["rate"] > r["random"]["rate"] for r in rows)
    verdict = ("oracle marginals BEAT random — the law's mechanism story is wrong, soften"
               if n_win >= 2 else
               "oracle marginals tie random — marginals are causally insufficient under "
               "coupling; the mechanism holds architecture-free")
    print(f"\noracle significantly beats random on {n_win}/{len(rows)} dims → {verdict}")

    payload = {
        "status": "ok",
        "method": "oracle-marginal proposal control on the coupled chained-bilinear family "
                  "(each coordinate drawn independently from the true per-variable marginal, "
                  "pooled from training-side instances; identical best-of-K polish + checker "
                  "and restart seeds as the random arm)",
        "config": {**vars(args), "ns": ns, "pool_seed0": POOL_SEED0,
                   "test_seed0": TEST_SEED0, "learned_source": str(COUPLED_JSON)},
        "seed_hygiene": {
            "pool_seeds": [POOL_SEED0, POOL_SEED0 + args.pool],
            "test_seeds": [TEST_SEED0, TEST_SEED0 + args.test],
            "overlap_instances": overlap,
            "note": "pool = first slice of the learned arm's own training range, so the "
                    "oracle sees the same data the learned model trained on; restart seeds "
                    "9000*s+nv shared with random_count (CRN)",
        },
        "rows": rows,
        "verdict": verdict,
        "wall_s": time.time() - t0,
    }
    out = Path("results/p_scaling/oracle_marginal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}; wall={payload['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
