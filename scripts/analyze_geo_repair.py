#!/usr/bin/env python3
"""Aggregate the R28 geometry-repair seed runs into one citable analysis artifact.

  python scripts/analyze_geo_repair.py \
      results/p_geo_repair/geo_repair_s11.json \
      results/p_geo_repair/geo_repair_s29.json \
      results/p_geo_repair/geo_repair_s47.json

Consumes the JSONs written by scripts/run_geo_repair.py and emits
results/p_geo_repair/analysis.json plus paste-ready text skeletons:

  * per-pool per-arm flip rates aggregated over optimization seeds
    (mean +/- population SD of each run's pool rate cells, R22 house shape);
  * Holm step-down correction over the declared six-comparison family
    (ranker vs restart_control/random/recipe_only/best_fixed/all_cos/probe)
    within each pool, on the primary seed's exact McNemar p-values recomputed
    from the stored win/loss counts; the other seeds ride along as raw-p
    robustness columns (data seed is shared, so the failure pools coincide);
  * the best label-vs-e2e agreement statistic the stored rows support.
    Rows carry the label COUNT (n_working) and per-arm e2e flips only — which
    candidates were measured working is not recoverable, so "ranker top1-hit
    under labels vs its e2e flip" cannot be computed here.  What can:
    existence-level agreement between the label stream (n_working > 0) and the
    e2e stream (the enumeration flip = some construction solved at full grade),
    plus a random-arm calibration against mean(n_working / n_candidates).
    Getting the top1-hit statistic needs a rerun that stores the per-candidate
    label vector (Packed.labels) in each row;
  * a RESULTS.md R28 skeleton and a marc.tex sentence block with every
    measured slot filled from the inputs and [SLOT: ...] markers for the
    judgment-shaped rest.  No number in either skeleton is invented.

Runs with however many of the three inputs have landed (>= 1)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.metrics import rate_cell

ARMS = ("ranker", "recipe_only", "random", "best_fixed", "all_cos",
        "restart_control", "restart_plus16", "restart_plus32",
        "probe", "enumeration")
BASELINES = ("restart_control", "random", "recipe_only", "best_fixed",
             "all_cos", "probe")


def _mcnemar(win: int, loss: int) -> float:
    d = win + loss
    if d == 0:
        return 0.5
    return sum(math.comb(d, j) for j in range(win, d + 1)) / (2.0 ** d)


def _holm(pvals: list) -> list:
    """Holm step-down adjusted p-values, returned in the input order (same as
    run_invention_eval.py)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _agg(values: list) -> dict:
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / len(values))
    return {"mean": m, "population_sd": sd,
            "min": min(values), "max": max(values), "values": values}


def _pm(a: dict) -> str:
    return f"{a['mean']:.3f} ± {a['population_sd']:.3f}"


def _pmx(a: dict) -> str:
    return f"{a['mean']:.3f} \\pm {a['population_sd']:.3f}"


def _fp(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def build_skeletons(out, kref, seeds, data_seed, primary_seed, n_seeds):
    pools = out["pools"]
    tr = pools["trained"]
    tf = pools.get("transfer")
    cost = out["cost"]
    la = out["label_agreement"]
    seeds_s = "/".join(str(s) for s in seeds)

    md = [f"## R28 · Geometry auxiliary-construction repair — "
          f"[SLOT: verdict headline — write it from the numbers below]",
          "",
          f"Population: 2D pruned distance-geometry chains the reference pipeline "
          f"(scipy LM, k_refine={kref}) fails as posed under two independent restart "
          f"streams; each arm's chosen construction gets one more reference solve on a "
          f"fresh common stream (`run_geo_repair.py`).  Data seed {data_seed} is shared "
          f"across optimization seeds {seeds_s}, so the failure pools coincide; flip "
          f"rates are mean ± population SD over the {n_seeds} seeds.  Labels and flips "
          f"are budget-relative measurements at the reference budget — NOT CAS "
          f"certificates; do not reuse R10's certificate language here.",
          ""]
    hdr = f"| arm | trained-k (N={tr['n_failures']})"
    sep = "|---|---:"
    if tf:
        hdr += f" | transfer-k (N={tf['n_failures']})"
        sep += "|---:"
    md += [hdr + " |", sep + "|"]
    for arm in ARMS:
        line = f"| {arm} | {_pm(tr['arms'][arm])}"
        if tf:
            line += f" | {_pm(tf['arms'][arm])}"
        md.append(line + " |")
    md += ["",
           f"Paired comparisons — exact McNemar on the common e2e stream, primary "
           f"seed {primary_seed}, Holm-corrected within each pool over the declared "
           f"{len(BASELINES)}-comparison family; the other seeds are raw-p robustness "
           f"columns:",
           ""]
    hdr = "| ranker vs | trained W/L | trained p (Holm)"
    sep = "|---|---:|---:"
    if tf:
        hdr += " | transfer W/L | transfer p (Holm)"
        sep += "|---:|---:"
    hdr += " | raw p, other seeds (trained) |"
    md += [hdr, sep + "|---|"]
    for b in BASELINES:
        t = tr["comparisons_holm"]["tests"][f"ranker_gt_{b}"]
        pr = t["primary"]
        line = (f"| {b} | {pr['ranker_only']}/{pr['baseline_only']} "
                f"| {_fp(pr['p_holm'])}")
        if tf:
            u = tf["comparisons_holm"]["tests"][f"ranker_gt_{b}"]["primary"]
            line += (f" | {u['ranker_only']}/{u['baseline_only']} "
                     f"| {_fp(u['p_holm'])}")
        rob = ", ".join(f"{_fp(r['p_one_sided_exact'])} (s{r['opt_seed']})"
                        for r in t["robustness"]) or "—"
        md.append(line + f" | {rob} |")
    md += ["",
           f"Cost (restarts per instance, mean over seeds): "
           f"ranker {cost['ranker']['restarts_per_instance']['mean']:.1f}, "
           f"probe {cost['probe']['restarts_per_instance']['mean']:.1f}, "
           f"restart_plus32 {cost['restart_plus32']['restarts_per_instance']['mean']:.1f}, "
           f"enumeration {cost['enumeration']['restarts_per_instance']['mean']:.1f}.",
           "",
           f"Label-vs-e2e agreement proxy (seed {primary_seed}, all "
           f"{la['n_rows']} test rows): \"label stream found ≥1 working "
           f"construction\" vs \"enumeration flipped on the e2e stream\" agree on "
           f"{la['agreement']['k']}/{la['agreement']['n']} = "
           f"{la['agreement']['rate']:.3f} "
           f"[{la['agreement']['ci95'][0]:.3f},{la['agreement']['ci95'][1]:.3f}] "
           f"(label+/e2e− {la['table']['label+_e2e-']}, label−/e2e+ "
           f"{la['table']['label-_e2e+']}).  Random-arm calibration: flip "
           f"{la['random_arm_calibration']['random_flip_cell']['rate']:.3f} vs "
           f"{la['random_arm_calibration']['expected_flip_rate_from_labels']:.3f} "
           f"expected from mean(n_working/n_candidates).  Per-candidate top1-hit "
           f"agreement is not derivable from the stored rows; rerun with the "
           f"per-candidate label vector stored per row to get it.",
           "",
           "Evidence: results/p_geo_repair/geo_repair_s{11,29,47}.json, "
           "results/p_geo_repair/analysis.json.",
           "[SLOT: PROVENANCE row — command, seeds, commit]"]

    def cite(pool, b):
        t = pool["comparisons_holm"]["tests"][f"ranker_gt_{b}"]
        rob = ", ".join(f"{_fp(r['p_one_sided_exact'])}"
                        for r in t["robustness"]) or "—"
        return f"p={_fp(t['primary']['p_holm'])} (raw p other seeds: {rob})"

    a = tr["arms"]
    tex = [
        "% --- R28 geometry construction repair (generated by "
        "analyze_geo_repair.py; verify before pasting) ---",
        f"On pruned distance-geometry chains the reference pipeline fails as posed "
        f"(two independent restart streams; $N={tr['n_failures']}$ trained-$k$"
        + (f", $N={tf['n_failures']}$ held-out-$k$ failures)," if tf else " failures),")
        + f" the construction ranker flips ${_pmx(a['ranker'])}$ of the trained-$k$ "
        f"pool at one augmented reference solve, against ${_pmx(a['restart_control'])}$ "
        f"for the matched-budget restart control and "
        f"${_pmx(a['restart_plus16'])}$/${_pmx(a['restart_plus32'])}$ at $+16$/$+32$ "
        f"restarts; the no-learning construction arms reach "
        f"${_pmx(a['best_fixed'])}$ (best fixed), ${_pmx(a['all_cos'])}$ (all cosine "
        f"lifts) and ${_pmx(a['recipe_only'])}$ (recipe-only ranker).",
        f"Exact paired McNemar tests on the common stream, Holm-corrected over the "
        f"six-comparison family, give {cite(tr, 'restart_control')} against the "
        f"restart control, {cite(tr, 'best_fixed')} against best-fixed and "
        f"{cite(tr, 'all_cos')} against all-cos on the primary seed "
        f"(seeds {seeds_s}, mean $\\pm$ population SD).",
        f"The probe control reaches ${_pmx(a['probe'])}$ at "
        f"{cost['probe']['restarts_per_instance']['mean']:.1f} restarts per instance "
        f"versus the ranker's {cost['ranker']['restarts_per_instance']['mean']:.1f}, "
        f"and enumeration bounds the ceiling at ${_pmx(a['enumeration'])}$ at "
        f"{cost['enumeration']['restarts_per_instance']['mean']:.1f}.",
    ]
    if tf:
        atf = tf["arms"]
        tex.append(
            f"On the held-out-$k$ pool the ranker flips ${_pmx(atf['ranker'])}$ "
            f"versus ${_pmx(atf['restart_control'])}$ for the restart control "
            f"({cite(tf, 'restart_control')}) and ${_pmx(atf['best_fixed'])}$ for "
            f"best-fixed ({cite(tf, 'best_fixed')}).")
    tex += [
        "Labels and flips are budget-relative measurements at the reference-solver "
        "budget, not certificates.",
        "[SLOT: claim sentence — write only after checking the "
        "ranker-vs-best\\_fixed/all\\_cos direction; if a fixed construction is "
        "competitive, report it as the honest ceiling.]",
    ]
    return "\n".join(md), "\n".join(tex)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R28 geo-repair multiseed analysis")
    ap.add_argument("paths", nargs="*",
                    default=[f"results/p_geo_repair/geo_repair_s{s}.json"
                             for s in (11, 29, 47)])
    ap.add_argument("--primary-seed", type=int, default=11,
                    help="opt_seed of the run whose McNemar p-values get Holm")
    ap.add_argument("--out", default="results/p_geo_repair/analysis.json")
    args = ap.parse_args(argv)

    runs, warnings = [], []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            warnings.append(f"input not landed yet, skipped: {p}")
            continue
        runs.append((path, json.loads(path.read_text())))
    if not runs:
        sys.exit("none of the input JSONs exist yet — nothing to analyze")
    for msg in warnings:
        print(msg, file=sys.stderr)

    if len({d["geo_repair_version"] for _, d in runs}) > 1:
        warnings.append("geo_repair_version differs across runs — do not pool")
    if len({d["config"]["seed"] for _, d in runs}) > 1:
        warnings.append("data seed differs across runs: failure pools are NOT "
                        "shared, robustness columns are not paired")

    prim_idx = next((i for i, (_, d) in enumerate(runs)
                     if d["config"]["opt_seed"] == args.primary_seed), None)
    if prim_idx is None:
        prim_idx, prim_reason = 0, (f"no run with opt_seed={args.primary_seed}; "
                                    "fell back to first input")
        warnings.append(prim_reason)
    else:
        prim_reason = f"opt_seed == {args.primary_seed}"
    prim_path, prim = runs[prim_idx]
    kref = prim["reference_solver"]["k_refine"]
    seeds = [d["config"]["opt_seed"] for _, d in runs]

    pool_names = [n for n in ("trained", "transfer")
                  if all(n in d["result"]["pools"] for _, d in runs)]
    pools_out = {}
    for pool in pool_names:
        prim_pool = prim["result"]["pools"][pool]
        arms = {}
        for arm in ARMS:
            vals = [d["result"]["pools"][pool][arm]["flip"]["rate"] for _, d in runs]
            arms[arm] = {**_agg(vals), "primary_cell": prim_pool[arm]["flip"]}
        names = [f"ranker_gt_{b}" for b in BASELINES]
        tests, pvals = {}, []
        for name in names:
            blk = prim_pool[name]["paired_mcnemar"]
            win, loss = blk["ranker_only"], blk["baseline_only"]
            p = _mcnemar(win, loss)
            if not math.isclose(p, blk["p_one_sided_exact"], rel_tol=1e-9):
                sys.exit(f"{pool}/{name}: recomputed McNemar p {p} disagrees with "
                         f"stored {blk['p_one_sided_exact']} — schema drift, stop")
            pvals.append(p)
            rob = []
            for i, (path, d) in enumerate(runs):
                if i == prim_idx:
                    continue
                b2 = d["result"]["pools"][pool][name]["paired_mcnemar"]
                rob.append({"opt_seed": d["config"]["opt_seed"], "path": str(path),
                            "ranker_only": b2["ranker_only"],
                            "baseline_only": b2["baseline_only"],
                            "p_one_sided_exact": _mcnemar(b2["ranker_only"],
                                                          b2["baseline_only"])})
            tests[name] = {"primary": {"ranker_only": win, "baseline_only": loss,
                                       "p_one_sided_exact": p},
                           "robustness": rob}
        for name, ph in zip(names, _holm(pvals)):
            tests[name]["primary"]["p_holm"] = ph
        pools_out[pool] = {
            "n_failures": prim_pool["n_failures"],
            "n_failures_per_seed": [d["result"]["pools"][pool]["n_failures"]
                                    for _, d in runs],
            "arms": arms,
            "comparisons_holm": {"method": "holm", "alpha": 0.05,
                                 "m": len(names),
                                 "primary_opt_seed": prim["config"]["opt_seed"],
                                 "family": names, "tests": tests},
        }

    cost = {arm: {k: _agg([d["result"]["cost"][arm][k] for _, d in runs])
                  for k in ("restarts_per_instance", "wall_s_per_instance")}
            for arm in ARMS}

    rows = prim["result"]["rows"]
    tab = {"label+_e2e+": 0, "label+_e2e-": 0, "label-_e2e+": 0, "label-_e2e-": 0}
    for r in rows:
        key = (f"label{'+' if r['n_working'] > 0 else '-'}"
               f"_e2e{'+' if r['enumeration']['flip'] else '-'}")
        tab[key] += 1
    n = len(rows)
    if n < 50:
        warnings.append(f"label-agreement sample is {n} rows (< 50) — quick run?")
    label_agreement = {
        "statistic": "existence-level agreement, primary seed: (n_working > 0) "
                     "[label stream] vs the enumeration flip [e2e stream: some "
                     "construction solved at full grade]",
        "why_not_top1": "rows store the label count and per-arm e2e flips only; "
                        "which candidates were measured working is not stored, so "
                        "the ranker's top1-hit under labels is not derivable.  To "
                        "get it, rerun run_geo_repair.py with the per-candidate "
                        "label vector (Packed.labels, keyed by construction name) "
                        "added to each row.",
        "n_rows": n, "sample_below_50": n < 50,
        "agreement": rate_cell(tab["label+_e2e+"] + tab["label-_e2e-"], n),
        "table": tab,
        "random_arm_calibration": {
            "random_flip_cell": rate_cell(
                sum(bool(r["random"]["flip"]) for r in rows), n),
            "expected_flip_rate_from_labels":
                sum(r["n_working"] / r["n_candidates"] for r in rows) / n,
            "note": "if the label stream transfers to the e2e stream, the random "
                    "arm's flip rate should track mean(n_working/n_candidates)",
        },
    }

    out = {
        "status": "ok",
        "method": "R28 multiseed aggregation, Holm-corrected primary-seed exact "
                  "McNemar (six-comparison family per pool), label-vs-e2e "
                  "agreement proxy",
        "geo_repair_version": prim["geo_repair_version"],
        "reference_solver": prim["reference_solver"],
        "inputs": [{"path": str(path), "opt_seed": d["config"]["opt_seed"],
                    "data_seed": d["config"]["seed"],
                    "geo_repair_version": d["geo_repair_version"],
                    "n_test_rows": len(d["result"]["rows"])}
                   for path, d in runs],
        "n_seeds": len(runs),
        "primary": {"path": str(prim_path),
                    "opt_seed": prim["config"]["opt_seed"],
                    "reason": prim_reason},
        "warnings": warnings,
        "pools": pools_out,
        "cost": cost,
        "workable_fraction": _agg([d["result"]["workable_fraction"]
                                   for _, d in runs]),
        "label_agreement": label_agreement,
    }
    md, tex = build_skeletons(out, kref, seeds, prim["config"]["seed"],
                              prim["config"]["opt_seed"], len(runs))
    out["skeletons"] = {"results_md": md, "marc_tex": tex}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    for pool in pool_names:
        po = pools_out[pool]
        line = " ".join(f"{a}={po['arms'][a]['mean']:.3f}"
                        for a in ("ranker", "restart_control", "best_fixed",
                                  "all_cos", "probe", "enumeration"))
        print(f"{pool} pool (n={po['n_failures']}, {len(runs)} seeds): {line}")
        holm = " ".join(
            f"{b}={_fp(po['comparisons_holm']['tests'][f'ranker_gt_{b}']['primary']['p_holm'])}"
            for b in BASELINES)
        print(f"{pool} Holm p (seed {prim['config']['opt_seed']}): {holm}")
    ag = label_agreement["agreement"]
    print(f"label-vs-e2e existence agreement: {ag['k']}/{ag['n']} = {ag['rate']:.3f}")
    for msg in warnings:
        print(f"warning: {msg}")
    print(f"wrote {outp}")
    print("\n---- RESULTS.md skeleton ----\n" + md)
    print("\n---- marc.tex block ----\n" + tex)


if __name__ == "__main__":
    main()
