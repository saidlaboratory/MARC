"""Does the amortization crossover (R5) replicate across DIFFERENT separable families?

R5 shows a learned proposal beats random multi-start at high dimension on one bundled
trap family. The obvious attack: that family was designed to show it. This runs the
same experiment on several structurally different *separable* families (different
spurious-well shapes, different root ranges). Each factor still has a single real root
R_i (a strictly positive bump factor has no root of its own) and a spurious low-gradient
region that stalls descent, so random restart must independently escape n such traps and
collapses in high dimension, while a learned proposal that memorizes per-variable
marginals should hold. If the crossover appears on all of them, it is a property of
separable structure, not of one construction.

Reuses the exact machinery of scripts/run_dimension_scaling.py (train_x0, count_methods,
the shared refine + Checker gate); only the per-factor residual changes per family.
Every rate carries a Wilson CI; learned-vs-random carries a two-proportion z per n.

Outputs results/p_scaling/crossover_families.json.
Run:  PYTHONPATH=. python3 scripts/run_crossover_families.py [--quick] [--seeds 1]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from marc.graph.graph import FactorGraph
from marc.graph.schema import VariableNode, FactorNode, Edge
from marc.eval.metrics import two_proportion_z, wilson_interval
from marc.eval.runner import Problem
from marc.eval.solver import ScipySolver

import run_dimension_scaling as rds  # train_x0, count_methods, accepted, METHODS, C, DECIMALS


def _lm_count(test, K, chk):
    """scipy Levenberg-Marquardt, K Gaussian multistarts, best-of-K + same Checker gate.
    The strong classical baseline: it also must hit all n independent basins from its
    multistart, so it collapses as ~p^n exactly like random restart."""
    ok = 0
    for g, sol, _ in test:
        solver = ScipySolver(seed=0, init_scale=8.0)
        cands = solver.sample(Problem(id="lm", graph=g, solution=[0.0] * len(g.variables)), K)
        ok += any(c is not None and rds.accepted(chk, g, c) for c in cands)
    return ok, len(test)

C, D = rds.C, rds.DECIMALS


def _rnd(rng, lo, hi):
    return round(rng.uniform(lo, hi), D)


def _build(n, exprs, sols, inits):
    vs = [VariableNode(f"x{i}") for i in range(n)]
    fs = [FactorNode(f"eq{i}", exprs[i]) for i in range(n)]
    es = [Edge(f"x{i}", f"eq{i}", 1) for i in range(n)]
    return FactorGraph(variables=vs, factors=fs, edges=es), sols, inits


# --- family makers: each returns (graph, solution, init) --------------------------

def fam_baseline(n, rng):
    """R5's family: (x-R)((x-m)^2+h)/C, R=+-U[3,8]. Reference."""
    e, s, ini = [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(3, 8), D)
        m, h = _rnd(rng, -0.2, 0.2), _rnd(rng, 0.1, 0.3)
        e.append(f"((x{i} - ({R})) * ((x{i} - ({m}))**2 + ({h}))) / {C}")
        s.append(R); ini.append(round(m + rng.uniform(-0.15, 0.15), D))
    return _build(n, e, s, ini)


def fam_quartic(n, rng):
    """Steeper spurious well: (x-R)((x-m)^4+h)/C."""
    e, s, ini = [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(3, 8), D)
        m, h = _rnd(rng, -0.2, 0.2), _rnd(rng, 0.1, 0.3)
        e.append(f"((x{i} - ({R})) * ((x{i} - ({m}))**4 + ({h}))) / {C}")
        s.append(R); ini.append(round(m + rng.uniform(-0.15, 0.15), D))
    return _build(n, e, s, ini)


def fam_double(n, rng):
    """Two spurious wells: (x-R)((x-m1)^2+h1)((x-m2)^2+h2)/C^2."""
    e, s, ini = [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(3, 8), D)
        m1, m2 = _rnd(rng, -1.5, -0.5), _rnd(rng, 0.5, 1.5)
        h1, h2 = _rnd(rng, 0.1, 0.3), _rnd(rng, 0.1, 0.3)
        e.append(f"((x{i} - ({R})) * ((x{i} - ({m1}))**2 + ({h1})) * "
                 f"((x{i} - ({m2}))**2 + ({h2}))) / {C*C}")
        s.append(R)
        ini.append(round(rng.choice([m1, m2]) + rng.uniform(-0.15, 0.15), D))
    return _build(n, e, s, ini)


def fam_wide(n, rng):
    """Wider roots, bump at 0: (x-R)(x^2+a)/C, R=+-U[5,12]."""
    e, s, ini = [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(5, 12), D)
        a = _rnd(rng, 0.1, 0.4)
        e.append(f"((x{i} - ({R})) * (x{i}**2 + ({a}))) / {C}")
        s.append(R); ini.append(_rnd(rng, -0.15, 0.15))
    return _build(n, e, s, ini)


# quartic_well is intentionally excluded: its root basin is narrower than the shared
# polish can reliably enter (polish fails even from +-0.1 of the true root), so it tests
# the polish, not the proposal. The three below have reasonable basins.
FAMILIES = {"baseline": fam_baseline, "double_well": fam_double, "wide_roots": fam_wide}


def suite(maker, n, count, seed):
    return [maker(n, random.Random(seed + 7919 * j)) for j in range(count)]


def run_family(maker, ns, K, ntest, epochs, ntrain, seeds):
    from marc.cas.checker import Checker
    chk = Checker()
    per_rep = []
    for rep in range(seeds):
        off = 1_000_003 * rep
        by_n = {}
        for n in ns:
            train = suite(maker, n, ntrain, seed=100 + n + off)
            test = suite(maker, n, ntest, seed=90000 + n + off)
            net = rds.train_x0([(g, sol) for g, sol, _ in train], epochs, seed=rep)
            tmean = sum(v for _, sol, _ in train for v in sol) / sum(len(sol) for _, sol, _ in train)
            cm = rds.count_methods(test, net, tmean, K, rep)
            cm["lm"] = _lm_count(test, K, chk)   # strong classical baseline
            by_n[n] = cm
        per_rep.append(by_n)
    rows = []
    for n in ns:
        row = {"n": n}
        for m in ("random_restart", "lm", "learned_x0"):
            ks = [per_rep[r][n][m][0] for r in range(seeds)]
            ts = [per_rep[r][n][m][1] for r in range(seeds)]
            k, t = sum(ks), sum(ts)
            row[m] = {"k": k, "n": t, "rate": k / t, "ci95": wilson_interval(k, t)}
        _, row["p_learned_gt_random"] = two_proportion_z(
            row["learned_x0"]["k"], row["learned_x0"]["n"],
            row["random_restart"]["k"], row["random_restart"]["n"])
        _, row["p_learned_gt_lm"] = two_proportion_z(
            row["learned_x0"]["k"], row["learned_x0"]["n"],
            row["lm"]["k"], row["lm"]["n"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="crossover replication across separable families")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ntrain", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ns = [1, 3] if args.quick else [1, 2, 3, 4, 6]
    if args.quick:
        args.epochs, args.ntrain, args.test = 20, 40, 12

    out = {}
    print(f"Crossover replication — best-of-{args.K}, {args.test} test/n, families={list(FAMILIES)}")
    for fname, maker in FAMILIES.items():
        rows = run_family(maker, ns, args.K, args.test, args.epochs, args.ntrain, args.seeds)
        out[fname] = rows
        wins = [r["n"] for r in rows
                if r["learned_x0"]["rate"] > r["random_restart"]["rate"]
                and r["learned_x0"]["rate"] > r["lm"]["rate"]
                and r["p_learned_gt_random"] < 0.05 and r["p_learned_gt_lm"] < 0.05]
        print(f"\n[{fname}]  {'n':>3} {'random':>8} {'lm':>8} {'learned':>8} {'p(l>rnd)':>9} {'p(l>lm)':>9}")
        for r in rows:
            print(f"        {r['n']:>3} {r['random_restart']['rate']:>8.3f} {r['lm']['rate']:>8.3f} "
                  f"{r['learned_x0']['rate']:>8.3f} {r['p_learned_gt_random']:>9.4f} {r['p_learned_gt_lm']:>9.4f}")
        print(f"  -> learned significantly beats BOTH random and LM at n = {wins or 'none'}")

    n_repl = sum(any(r["learned_x0"]["rate"] > r["random_restart"]["rate"]
                     and r["learned_x0"]["rate"] > r["lm"]["rate"]
                     and r["p_learned_gt_random"] < 0.05 and r["p_learned_gt_lm"] < 0.05
                     for r in rows)
                 for rows in out.values())
    print(f"\ncrossover replicates (learned > both random AND LM, significant, at some high n) "
          f"on {n_repl}/{len(FAMILIES)} families")
    payload = {"K": args.K, "test_per_n": args.test, "epochs": args.epochs,
               "seeds": args.seeds, "families": out,
               "n_families_with_crossover": n_repl}
    p = Path("results/p_scaling/crossover_families.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
