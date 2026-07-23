#!/usr/bin/env python3
"""Recompute every headline number from the committed result JSONs and check it
still matches the paper. Two-sided: a check fails if the JSON drifts from the
value we recorded (bad rerun / stale cache) OR if the value no longer appears in
marc_aaai.tex (paper edited away from the data). This is the guard we lacked
every time a number moved across data versions.

  python3 scripts/verify_paper_numbers.py        # exits non-zero on any drift

Add a row to CHECKS when a new cited number lands; the coverage count is printed
so gaps are visible rather than silent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = (ROOT / "paper/tex/marc_aaai.tex").read_text()


def cell(d, *path):
    for k in path:
        d = d[k]
    return d


def _load(rel):
    return json.loads((ROOT / rel).read_text())


# (label, json path, getter -> float, expected value, decimals, token-in-tex or None)
CHECKS = [
    # R10 repair ranker (Data Version 8)
    ("R10 nonlinear full", "results/p_repair/nonlinear_balanced_full_paired.json",
     lambda d: cell(d, "result", "full", "invention", "rate"), 0.997, 3, "0.997"),
    ("R10 nonlinear random", "results/p_repair/nonlinear_balanced_full_paired.json",
     lambda d: cell(d, "result", "random", "invention", "rate"), 0.236, 3, "0.236"),
    ("R10 nonlinear control", "results/p_repair/nonlinear_balanced_full_paired.json",
     lambda d: cell(d, "result", "control", "invention", "rate"), 0.333, 3, "0.333"),
    ("R10 nonlinear multiseed mean", "results/p_repair/nonlinear_multiseed.json",
     lambda d: cell(d, "aggregate", "full", "mean"), 0.982, 3, "0.982"),
    ("R10 nonlinear multiseed sd", "results/p_repair/nonlinear_multiseed.json",
     lambda d: cell(d, "aggregate", "full", "population_sd"), 0.006, 3, "0.006"),
    ("R10 linear all-pattern", "results/p_repair/random_support_holdout_shared.json",
     lambda d: cell(d, "result", "full", "invention", "rate"), 0.339, 3, "0.339"),
    ("R10 linear shared holdout N=400", "results/p_repair/random_support_holdout_shared.json",
     lambda d: cell(d, "result", "per_family", "aux_required:shared", "full", "rate"),
     0.380, 3, "0.380"),

    # R15 dimension-scaling law inputs (unified-v2, single seed = the cited cells)
    ("R15 learned n=3", "results/p_scaling/scaling.json",
     lambda d: d["rows"][2]["learned_x0"]["rate"], 0.975, 3, "0.975"),
    ("R15 random n=2", "results/p_scaling/scaling.json",
     lambda d: d["rows"][1]["random_restart"]["rate"], 0.725, 3, "0.725"),
    ("R15 random n=3", "results/p_scaling/scaling.json",
     lambda d: d["rows"][2]["random_restart"]["rate"], 0.075, 3, "0.075"),
    ("R15 learned n=6", "results/p_scaling/scaling.json",
     lambda d: d["rows"][4]["learned_x0"]["rate"], 0.250, 3, "0.250"),

    # R28 geometry construction repair (v3 + concentration control)
    ("R28 ranker trained", "results/p_geo_repair/analysis_v3.json",
     lambda d: cell(d, "pools", "trained", "arms", "ranker", "mean"), 0.246, 3, "0.246"),
    ("R28 random trained", "results/p_geo_repair/analysis_v3.json",
     lambda d: cell(d, "pools", "trained", "arms", "random", "mean"), 0.185, 3, "0.185"),
    ("R28 restart_control trained", "results/p_geo_repair/analysis_v3.json",
     lambda d: cell(d, "pools", "trained", "arms", "restart_control", "mean"), 0.270, 3, "0.270"),
    ("R28 probe trained", "results/p_geo_repair/analysis_v3.json",
     lambda d: cell(d, "pools", "trained", "arms", "probe", "mean"), 0.698, 3, "0.698"),
    ("R28 enumeration trained", "results/p_geo_repair/analysis_v3.json",
     lambda d: cell(d, "pools", "trained", "arms", "enumeration", "mean"), 0.692, 3, "0.692"),
    ("R28 cross-fit screen trained", "results/p_geo_repair/probe_concentration.json",
     lambda d: cell(d, "pools", "trained", "crossfit_top1_at_kref", "rate"), 0.199, 3, "0.199"),

    # R30 real-systems repair. token=None: filed in RESULTS/PROVENANCE, not yet in the
    # 7-page tex (pending #122 integration) — check JSON stability, flip the token on
    # once the R30 paragraph lands in marc_aaai.tex.
    ("R30 trilat ceiling", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "trilat_far")["ceiling"]["rate"],
     1.000, 3, None),
    ("R30 trilat restart+4V", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "trilat_far")["restart_matched"]["rate"],
     0.379, 3, None),
    ("R30 conic ceiling", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "conic_ghost")["ceiling"]["rate"],
     1.000, 3, None),
]


def main() -> int:
    fails = []
    for label, path, getter, expected, places, token in CHECKS:
        try:
            got = round(float(getter(_load(path))), places)
        except Exception as exc:  # missing key / file = a real drift, report it
            fails.append(f"{label}: could not read {path} ({exc})")
            continue
        if abs(got - expected) > 0.5 * 10 ** (-places):
            fails.append(f"{label}: JSON has {got}, recorded {expected} ({path})")
        elif token is not None and token not in TEX:
            fails.append(f"{label}: {token} absent from marc_aaai.tex (paper drifted from data)")
        else:
            print(f"ok   {label:32} {got:.{places}f}")
    print(f"\n{len(CHECKS) - len(fails)}/{len(CHECKS)} checks passed")
    for f in fails:
        print(f"FAIL {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
