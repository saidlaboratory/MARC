#!/usr/bin/env python3
"""Seed robustness for R30 (real-systems construction repair).

The citable R30 run measures one population per class. This wrapper repeats the
whole protocol -- generation, two-stream failure selection, held-out construction
selection, budget-matched restart controls -- on independent seed bases, so the
headline "a single derived construction repairs the failure population" is
reported with sampling variance rather than from one draw. This matches the bar
run_repair_multiseed.py sets for the R10 ranker.

Each seed base shifts every class base by the same offset; bases are separated by
1e6, far beyond both --n and the 40k inter-class stride, so the populations are
disjoint. Within a run the stream hygiene is the pilot's (selection=seed,
second=seed+SALT, grading=seed+3*SALT).

Run:  PYTHONPATH=. python3 scripts/run_real_repair_multiseed.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from pilot_real_repair import K_REF, TOL, fmt  # noqa: F401  (fmt used in printing)
from run_real_repair import CLASSES, held_out_selection, run_class

from marc.eval.metrics import rate_cell
from marc.structure.invention_data import REFERENCE_SOLVER

DEFAULT_SEEDS = (20260722, 21260722, 22260722)


def _rate(cell):
    return cell["rate"] if isinstance(cell, dict) else cell


def main() -> int:
    ap = argparse.ArgumentParser(description="R30 seed robustness")
    ap.add_argument("--n", type=int, default=200, help="instances per class per seed")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--out", default="results/p_real_repair/real_repair_multiseed.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"R30 multiseed: reference = LM k={K_REF}, acceptance max|r|<{TOL} on original "
          f"factors; seed bases {seeds}\n")

    t0 = time.time()
    runs = []
    for seed in seeds:
        per_class = {}
        for idx, (cname, gen) in enumerate(CLASSES):
            rep = run_class(cname, gen, seed + 40_000 * idx, args.n)
            if rep["n_fail"]:
                sel_cell, per_fold = held_out_selection(rep["rows"])
                rep["held_out_selection"] = sel_cell
                rep["held_out_folds"] = per_fold
            # drop per-instance rows: the aggregate is the artifact, and the rows
            # are ~4k entries per class that no cited number reads
            rep.pop("rows", None)
            per_class[cname] = rep
            tail = ""
            if rep.get("held_out_selection"):
                tail = (f" ceiling={fmt(rep['ceiling'])}"
                        f" held-out-sel={fmt(rep['held_out_selection'])}"
                        f" restart+KV={fmt(rep['restart_matched'])}")
            print(f"[seed {seed}] {cname:14} fail={fmt(rep['fail'])} "
                  f"n_fail={rep['n_fail']}{tail}", flush=True)
        runs.append({"seed": seed, "classes": per_class})
        print()

    # aggregate only over classes that produce a failure population in every run;
    # a class that never bites is a documented negative, not a zero to average in
    biting = [c for c, _ in CLASSES
              if all(r["classes"][c]["n_fail"] for r in runs)]
    negatives = [c for c, _ in CLASSES
                 if not any(r["classes"][c]["n_fail"] for r in runs)]

    aggregate = {}
    for cname in biting:
        cells = [r["classes"][cname] for r in runs]
        picks = sorted({v["selected"]
                        for c in cells for v in c["held_out_folds"].values()})
        agg = {"n_fail": [c["n_fail"] for c in cells],
               "constructions_selected": picks,
               "selection_unanimous": len(picks) == 1}
        for arm in ("fail", "ceiling", "held_out_selection", "restart4",
                    "restart_matched"):
            vals = [_rate(c[arm]) for c in cells]
            agg[arm] = {"per_seed": [round(v, 4) for v in vals],
                        "mean": statistics.fmean(vals),
                        "population_sd": statistics.pstdev(vals)}
        # pooled held-out selected rate across every fold of every seed
        hits = sum(v["hits"] for c in cells for v in c["held_out_folds"].values())
        tries = sum(v["n"] for c in cells for v in c["held_out_folds"].values())
        agg["held_out_pooled"] = rate_cell(hits, tries)
        aggregate[cname] = agg

    for cname, agg in aggregate.items():
        print(f"=== {cname} (n_fail per seed {agg['n_fail']}) ===")
        print(f"  held-out selected  {agg['held_out_selection']['mean']:.3f} "
              f"+- {agg['held_out_selection']['population_sd']:.3f}"
              f"   pooled {fmt(agg['held_out_pooled'])}")
        print(f"  restart +K*V       {agg['restart_matched']['mean']:.3f} "
              f"+- {agg['restart_matched']['population_sd']:.3f}")
        print(f"  construction       {agg['constructions_selected']} "
              f"(unanimous={agg['selection_unanimous']})")

    payload = {
        "experiment": "real_systems_construction_repair_multiseed",
        "issue": 124, "arc": "R30",
        "reference_solver": dict(REFERENCE_SOLVER), "tol": TOL,
        "protocol": "independent seed bases (separated by 1e6); within each run the "
                    "pilot stream hygiene applies (selection=seed, second=seed+SALT, "
                    "grading=seed+3*SALT) and construction selection is held out by "
                    "seed parity, so no failure instance is both chooser and score",
        "config": vars(args), "seeds": seeds,
        "biting_classes": biting, "documented_negatives": negatives,
        "aggregate": aggregate, "runs": runs,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({payload['wall_clock_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
