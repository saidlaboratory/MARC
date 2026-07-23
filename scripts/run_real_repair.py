#!/usr/bin/env python3
"""R30 protocol: derived-construction repair on the named real systems (citable).

Promotes scripts/pilot_real_repair.py to a results-grade run for the #124 external
anchor ("the construction-repair result is not an artifact of benchmarks we built").
Reuses the pilot's validated generators, two-stream failure selection, and
budget-matched restart controls verbatim -- this wrapper only adds what a citable
claim needs beyond the pilot:

  * a HELD-OUT construction-selection arm. The pilot's ``best_single`` is picked
    in-sample (optimistic). Here the failure pool is split by seed parity; the
    construction with the most flips on the selection half is scored on the
    disjoint evaluation half. That number generalizes -- it forecloses the "you
    enumerated V constructions and reported the luckiest" critique -- and is
    reported alongside the non-circular ceiling-vs-enumeration-budget comparison.
  * a seed_hygiene block (the pilot's streams are salt-separated: selection seed,
    +SALT second stream, +3*SALT common grading stream; selection and grading
    never share a draw).
  * provenance + a results/ path (the pilot writes /tmp only).

The headline citable statistic is the ceiling (any derived construction solves)
vs. the enumeration-budget-matched restart control at equal solver calls, plus the
held-out selected-construction rate. Numbers go in as-is (do-not-spin rule):
classes with no failure population are reported as documented negatives.

Run:  PYTHONPATH=. python3 scripts/run_real_repair.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pilot_real_repair import (  # validated pilot machinery
    CLASSES,
    K_REF,
    SALT,
    TOL,
    _mcnemar,
    fmt,
    run_class,
)
from marc.eval.metrics import rate_cell
from marc.structure.invention_data import REFERENCE_SOLVER


def held_out_selection(rows):
    """Honest generalizing selected-construction rate: pick the best construction
    on the seed-even half, score it on the seed-odd half (and symmetrically), then
    pool. Returns (rate_cell, per_fold) or None when a fold is empty."""
    folds = {0: [r for r in rows if (r["seed"] % 2) == 0],
             1: [r for r in rows if (r["seed"] % 2) == 1]}
    if not folds[0] or not folds[1]:
        return None, {}
    hits = tries = 0
    per_fold = {}
    for train_key, eval_key in ((0, 1), (1, 0)):
        train, ev = folds[train_key], folds[eval_key]
        counts = {}
        for r in train:
            for name, ok in r["worked"].items():
                counts.setdefault(name, 0)
                counts[name] += int(ok)
        pick = max(counts, key=lambda k: counts[k]) if counts else None
        fold_hits = sum(int(r["worked"].get(pick, False)) for r in ev)
        hits += fold_hits
        tries += len(ev)
        per_fold[f"train{train_key}_eval{eval_key}"] = {
            "selected": pick, "hits": fold_hits, "n": len(ev)}
    return rate_cell(hits, tries), per_fold


def main():
    ap = argparse.ArgumentParser(description="R30 real-systems construction repair (citable)")
    ap.add_argument("--n", type=int, default=200, help="instances per class")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="results/p_real_repair/real_repair.json")
    args = ap.parse_args()

    print(f"R30: reference = LM k={K_REF}, acceptance max|r|<{TOL} on original factors; "
          f"two-stream failure selection; held-out construction selection by seed parity\n")
    reports = []
    for idx, (cname, gen) in enumerate(CLASSES):
        rep = run_class(cname, gen, args.seed + 40_000 * idx, args.n)
        if rep["n_fail"]:
            sel_cell, per_fold = held_out_selection(rep["rows"])
            rep["held_out_selection"] = sel_cell
            rep["held_out_folds"] = per_fold
        reports.append(rep)
        tail = ""
        if rep.get("held_out_selection"):
            tail = (f" ceiling={fmt(rep['ceiling'])} restart+V={fmt(rep['restart_matched'])}"
                    f" held-out-sel={fmt(rep['held_out_selection'])}"
                    f" p={rep['mcnemar_ceiling_vs_matched']:.4f}")
        print(f"[{cname}] fail={fmt(rep['fail'])} n_fail={rep['n_fail']} V={rep['vocab_size']}"
              f"{tail}", flush=True)

    biting = [r["class"] for r in reports
              if r["n_fail"] and r["fail"]["rate"] >= 0.2
              and r["mcnemar_ceiling_vs_matched"] < 0.05]
    negatives = [r["class"] for r in reports if not r["n_fail"]]
    verdict = (
        f"POSITIVE external anchor: {len(biting)} of {len(reports)} real-system classes "
        f"({biting}) produce a >=20% two-stream failure population that derived "
        f"constructions clear at a rate the enumeration-budget-matched restart control "
        f"cannot (McNemar p<0.05). Documented negatives (no failure population): "
        f"{negatives or 'none'}."
        if len(biting) >= 2 else
        f"Insufficient: only {biting} bite; file as one-paragraph negative per #124.")

    print("\n" + verdict)

    hyg = {
        "streams": "selection=seed, second=seed+SALT, grading/e2e=seed+3*SALT; "
                   "held-out selection splits the grading pool by seed parity, so the "
                   "construction chooser and its evaluation never share a failure instance",
        "salt": SALT,
        "class_seed_bases": {c: args.seed + 40_000 * i for i, (c, _) in enumerate(CLASSES)},
        "overlap_instances": 0,
    }
    payload = {
        "experiment": "real_systems_construction_repair", "issue": 124, "arc": "R30",
        "reference_solver": dict(REFERENCE_SOLVER), "tol": TOL,
        "acceptance": "numeric max|r|<tol on ORIGINAL factors only; aux factors join the "
                      "least-squares but never acceptance",
        "headline": "ceiling (any derived construction) and held-out selected construction "
                    "vs enumeration-budget-matched restart at equal solver calls",
        "config": vars(args), "seed_hygiene": hyg,
        "biting_classes": biting, "documented_negatives": negatives,
        "verdict": verdict, "classes": reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
