#!/usr/bin/env python3
"""Pilot: does construction repair hold in the native-3D DMDGP setting?

The 2D result (R28) was a closed negative under two-stream selection. This takes
the same machinery to 3D pruned chains (marc.data.geometry.make_pruned_chain_3d,
marc.structure.geo_repair3d) with the SAME discipline that caught us in 2D:
two-stream failure selection FIRST, then the restart-scaling curve (+4/+16/+32)
measured BEFORE any construction claim. If restart scaling matches the
construction ceiling, that is the result ("the trap holds in 3D"); if the
ceiling clears budget-matched restarts, 3D is where construction repair bites.

Pilot: prints the table, writes results/p_geo_repair/pilot3d.json.
Run:  PYTHONPATH=. python3 scripts/pilot_geo_repair3d.py [--n 60 --ks 6,8]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.data.geometry import make_pruned_chain_3d
from marc.eval.metrics import rate_cell
from marc.structure.geo_repair import STREAM_SALT, solve_graph
from marc.structure.geo_repair3d import construction_vocabulary_3d
from marc.structure.invention_data import REFERENCE_SOLVER

K_REF = REFERENCE_SOLVER["k_refine"]


def _mcnemar(win, loss):
    d = win + loss
    return 0.5 if d == 0 else sum(math.comb(d, j) for j in range(win, d + 1)) / (2.0 ** d)


def run_k(k, base, n):
    fails = []
    for t in range(n):
        s = base + t
        g, _sol, gv = make_pruned_chain_3d(k, random.Random(s))
        if solve_graph(g, seed=s) or solve_graph(g, seed=s + STREAM_SALT):
            continue                                   # two-stream selection
        fails.append((s, g, construction_vocabulary_3d(k, gv)))

    rows = []
    for s, g, vocab in fails:
        e2e = s + 3 * STREAM_SALT                       # fresh common stream, all arms
        worked = [solve_graph(c.apply(g), seed=e2e) for c in vocab]
        order = list(range(len(vocab)))
        random.Random(e2e + 11).shuffle(order)
        enum = next((True for j in order if solve_graph(vocab[j].apply(g), seed=e2e)), False)
        rows.append({
            "seed": s, "V": len(vocab), "n_working": sum(worked),
            "ceiling": any(worked),
            "restart4": solve_graph(g, seed=e2e, k_restarts=K_REF),
            "restart16": solve_graph(g, seed=e2e, k_restarts=16),
            "restart32": solve_graph(g, seed=e2e, k_restarts=32),
            "enumeration": enum,
        })
    nf = len(rows)
    rep = {"k": k, "n": n, "n_fail": nf, "fail": rate_cell(nf, n)}
    if nf:
        for arm in ("ceiling", "restart4", "restart16", "restart32", "enumeration"):
            rep[arm] = rate_cell(sum(r[arm] for r in rows), nf)
        rep["mcnemar_ceiling_vs_restart32"] = _mcnemar(
            sum(r["ceiling"] and not r["restart32"] for r in rows),
            sum(r["restart32"] and not r["ceiling"] for r in rows))
        rep["rows"] = rows
    return rep


def fmt(c):
    return f"{c['rate']:.2f}[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]"


def main():
    ap = argparse.ArgumentParser(description="3D DMDGP construction-repair pilot")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ks", default="6,8")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="results/p_geo_repair/pilot3d.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    print(f"3D DMDGP pilot — LM k={K_REF}, two-stream failure selection, "
          f"CRN grading stream seed+3*SALT")
    t0 = time.time()
    reps = []
    for idx, k in enumerate(ks):
        rep = run_k(k, args.seed + 40_000 * idx, args.n)
        reps.append(rep)
        if rep["n_fail"]:
            print(f"[k={k}] fail={fmt(rep['fail'])} n_fail={rep['n_fail']} | "
                  f"ceiling={fmt(rep['ceiling'])} restart+4={fmt(rep['restart4'])} "
                  f"+16={fmt(rep['restart16'])} +32={fmt(rep['restart32'])} "
                  f"enum={fmt(rep['enumeration'])} | McNemar ceil>restart32 "
                  f"p={rep['mcnemar_ceiling_vs_restart32']:.4f}", flush=True)
        else:
            print(f"[k={k}] fail={fmt(rep['fail'])} n_fail=0 (no population)", flush=True)

    biting = [r["k"] for r in reps if r["n_fail"]
              and r["ceiling"]["rate"] - r["restart32"]["rate"] > 0.1
              and r["mcnemar_ceiling_vs_restart32"] < 0.05]
    verdict = ("construction ceiling clears +32 restarts on k=" + str(biting)
               + " — the mechanism bites in 3D" if biting else
               "construction ceiling matches restart scaling — the 2D trap holds in 3D")
    print(f"\n{verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "reference_solver": dict(REFERENCE_SOLVER), "config": vars(args),
        "verdict": verdict, "ks": reps, "wall_s": time.time() - t0}, indent=2))
    print(f"wrote {args.out}; wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
