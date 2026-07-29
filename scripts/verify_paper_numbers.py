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
# Body + technical appendix: cited numbers live in both, and a number moved from one to the
# other must not read as drift.
TEX = "\n".join((ROOT / p).read_text() for p in
                ("paper/tex/marc_aaai.tex", "paper/tex/marc_aaai_appendix.tex"))


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

    # R30 real-systems repair, single citable run. The paragraph cites the three-seed
    # aggregate below, so these rows guard JSON stability only (their rates also appear
    # in the multiseed per_seed lists) — no token.
    ("R30 trilat ceiling", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "trilat_far")["ceiling"]["rate"],
     1.000, 3, None),
    ("R30 trilat restart+4V", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "trilat_far")["restart_matched"]["rate"],
     0.382, 3, None),
    # 3-seed dimension-scaling addendum (cited in Limitations + scaling caption)
    ("R15b learned n=6 3-seed mean", "results/p_scaling/scaling_3seed.json",
     lambda d: next(r for r in d["rows"] if r["n"] == 6)["learned_x0"]["seed_mean"],
     0.983, 3, "0.983"),
    ("R15b learned n=4 3-seed mean", "results/p_scaling/scaling_3seed.json",
     lambda d: next(r for r in d["rows"] if r["n"] == 4)["learned_x0"]["seed_mean"],
     0.658, 3, "0.658"),
    # entrapment 95% CI half-width (was mis-stated 0.086 until Jul 29)
    ("R5 entrapment ci95", "results/p1_entrapment/summary.json",
     lambda d: d["entrapment_reduction_ci95"], 0.109, 3, "0.109"),
    ("R30 conic ceiling", "results/p_real_repair/real_repair.json",
     lambda d: next(c for c in d["classes"] if c["class"] == "conic_ghost")["ceiling"]["rate"],
     1.000, 3, None),

    # R30 three-seed aggregate — these are the numbers the external-anchor paragraph cites
    ("R30 trilat fail mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "fail", "mean"), 0.848, 3, "0.848"),
    ("R30 trilat fail sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "fail", "population_sd"), 0.020, 3, "0.020"),
    ("R30 trilat held-out sel", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "held_out_selection", "mean"), 1.000, 3, "1.000"),
    ("R30 trilat held-out N", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "held_out_pooled", "n"), 509, 0, "509"),
    ("R30 trilat restart mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "restart_matched", "mean"), 0.433, 3, "0.433"),
    ("R30 trilat restart sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "restart_matched", "population_sd"),
     0.049, 3, "0.049"),
    ("R30 conic fail mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "fail", "mean"), 0.263, 3, "0.263"),
    ("R30 conic fail sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "fail", "population_sd"), 0.015, 3, "0.015"),
    ("R30 conic held-out N", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "held_out_pooled", "n"), 158, 0, "158"),
    ("R30 conic restart mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "restart_matched", "mean"), 0.114, 3, "0.114"),
    ("R30 conic restart sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "restart_matched", "population_sd"),
     0.011, 3, "0.011"),
    # own-budget restart control (Appendix Table 5): the reference solver rerun on the
    # unrepaired system at its own K, before the enumeration budget is matched
    ("R30 trilat restart own mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "restart4", "mean"), 0.079, 3, "0.079"),
    ("R30 trilat restart own sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "trilat_far", "restart4", "population_sd"),
     0.017, 3, "0.017"),
    ("R30 conic restart own mean", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "restart4", "mean"), 0.026, 3, "0.026"),
    ("R30 conic restart own sd", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: cell(d, "aggregate", "conic_ghost", "restart4", "population_sd"),
     0.010, 3, "0.010"),
    # the anchor's central claim: the selected construction is unanimous across all six
    # folds, which is what forecloses "the luckiest of V"
    ("R30 selection unanimous", "results/p_real_repair/real_repair_multiseed.json",
     lambda d: float(all(cell(d, "aggregate", c, "selection_unanimous")
                         for c in ("trilat_far", "conic_ghost"))), 1.0, 1, None),

    # R23b operator-mask ablation. Regenerated this pass: the artifact was recorded in
    # RESULTS.md but had never been committed, so nothing could check it.
    # NB: this arm is not bit-reproducible at fixed seed (0.978/0.981/0.986 observed across
    # reruns of the identical command), so it is checked to 2 decimals and the paper cites
    # the Wilson interval rather than the third decimal.
    ("R23b opmask nonlinear masked", "results/p_repair/nonlinear_opmask_ablation.json",
     lambda d: cell(d, "result", "full", "invention", "rate"), 0.98, 2, "0.978"),

    # R3 hybrid battery (Appendix Table 4). Added after the paper's BilinearProduct cell
    # (0.717 / "random wins") was found to disagree with the committed artifact (0.683 / tie);
    # the table is now generated from this JSON, and these rows keep it that way.
    ("R3 hybrid random BilinearSystem", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][0]["random_restart"]["rate"], 0.550, 3, "0.550"),
    ("R3 hybrid random BilinearProduct", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][1]["random_restart"]["rate"], 0.683, 3, "0.683"),
    ("R3 hybrid random QuadraticSystem", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][2]["random_restart"]["rate"], 0.683, 3, "0.683"),
    ("R3 hybrid random CircleLine", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][3]["random_restart"]["rate"], 0.200, 3, "0.200"),
    ("R3 hybrid learned BilinearProduct", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][1]["learned_hybrid"]["rate"], 0.683, 3, "0.683"),
    ("R3 hybrid learned CircleLine", "results/p_hard/hard_eval.json",
     lambda d: d["rows"][3]["learned_hybrid"]["rate"], 0.000, 3, "0.000"),
    ("R3 hybrid LM saturates all four", "results/p_hard/hard_eval.json",
     lambda d: float(all(r["lm"]["rate"] == 1.0 for r in d["rows"])), 1.0, 1, "1.000"),

    # R26 real systems, per-system reachability under the gradient polish (Table 3)
    ("R26 q 2R inverse kinematics", "results/p_real/real_systems.json",
     lambda d: next(r for r in d["rows"]
                    if r["name"] == "inverse_kinematics_2r")["q_single_start"]["rate"],
     0.98, 2, "0.98"),
    ("R26 q cyclic-4", "results/p_real/real_systems.json",
     lambda d: next(r for r in d["rows"] if r["name"] == "cyclic4")["q_single_start"]["rate"],
     0.38, 2, "0.38"),
    ("R26 LM solves all eight", "results/p_real/real_systems.json",
     lambda d: float(cell(d, "solved_counts", "lm")), 8, 0, None),
    ("R26 random restart solves four", "results/p_real/real_systems.json",
     lambda d: float(cell(d, "solved_counts", "random_restart")), 4, 0, None),
    ("R26 Langevin solves one", "results/p_real/real_systems.json",
     lambda d: float(cell(d, "solved_counts", "langevin")), 1, 0, None),
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
