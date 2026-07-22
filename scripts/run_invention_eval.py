"""End-to-end invention eval (U5/W1): does the trained policy's picked augmentation
actually make the graph solvable — on seeds provably disjoint from model selection?

Protocol (W1 overhaul):
  * Multi-seed test set: eval seeds ``seed_base + j*SEED_STRIDE`` (default 5 seeds
    from 900000), each generating ``--n`` instances. Headline rates are POOLED
    counts across seeds; per-seed descriptive counts live in ``per_seed``.
  * Seed hygiene: at checkpoint load, every eval seed range is asserted disjoint
    from the training and validation (model-selection) seed ranges recorded in —
    or reconstructed from — ``train_config``. Violation exits 2 unless
    ``--allow-seed-overlap``.
  * Enumeration baseline: the linear menu is exactly solvable by trying each
    candidate; the ``arms.enumeration`` block runs it FIRST per instance on a cold
    solver cache with honest wall-clock, so the policy's claim is amortization,
    not capability.
  * Holm step-down correction over the declared comparison family
    (``comparisons_holm``).

Missing/unloadable checkpoint -> {"status": "skipped"}, exit 0.

Usage:
    python3 scripts/run_invention_eval.py --ckpt checkpoints/structure_policy.pt \
        --out results/p5_invention/invention.json [--n 100] [--k-refine 4] \
        [--data toys|aux_required] [--K 4] [--eval-seeds 5] [--seed-base 900000] \
        [--families toy1,toy2] [--allow-seed-overlap]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import marc.structure.invention_data as invention_data
from marc.cas.checker import Checker
from marc.eval.metrics import rate_cell, two_proportion_z
from marc.eval.runner import Problem
from marc.eval.solver import load_solver
from marc.structure.invention_data import FAMILIES, InventionInstance, make_dataset
from marc.structure.policy import (
    StructurePolicy,
    chosen_candidate,
    predicted_pin,
    reverse_sample,
)

TEST_SEED_BASE = 900000   # seed-space contract v1 — must match scripts/train_structure_policy.py
SEED_STRIDE = 1000        # seed-space contract v1 — must match scripts/train_structure_policy.py
LEGACY_VAL_OFFSET = 500000  # seed-space contract v1 — must match scripts/train_structure_policy.py
LEGACY_VAL_SIZE = 50        # seed-space contract v1 — must match scripts/train_structure_policy.py

# Reference solver for the eval's solve-rate arms. scipy Levenberg–Marquardt (analytic
# Jacobian, k Gaussian multistarts): it matches aux_required's *exact* solvability
# certificate — the gradient-descent `refine` solver is too weak for the nonlinear
# augmented systems (gold-oracle positive control collapsed to ~0.02, floor-ing every
# solve rate), while LM solves the certified-solvable systems 100% (gold-oracle ~1.0),
# so solve-rate comparisons across arms become meaningful. Invention-accuracy (pick==gold)
# is independent of this choice. The dict is OWNED by invention_data so distractor
# certification and arm grading can never diverge again.
REFERENCE_SOLVER = invention_data.REFERENCE_SOLVER

#: capability probe P4: sibling data units may widen the source list.
SOURCES = tuple(getattr(invention_data, "SOURCES", ("toys", "aux_required")))


def _hard_negative_idxs(inst: InventionInstance) -> set[int]:
    """Indices of gold-structure/wrong-value distractors. Scalar-pin menus have
    one; exchangeable expression menus coefficient-match EVERY distractor, so
    confusion must be counted against the whole set, not the first hit."""
    gold = inst.candidates[inst.gold_idx]
    return {
        j for j, c in enumerate(inst.candidates)
        if j != inst.gold_idx and c.insert_coeffs == gold.insert_coeffs
    }


def _holm(pvals: list) -> list:
    """Holm step-down adjusted p-values, returned in the input order.

    Ascending order p_(1) <= ... <= p_(m): p_holm(i) = max_{j<=i} min(1, (m-j+1)*p_(j)).
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _check_seed_hygiene(train_config: dict, data: str, eval_seeds: list,
                        n: int, allow_overlap: bool) -> dict:
    """Assert every eval seed range is disjoint from the train/val (model-selection)
    seed ranges. Violation -> SystemExit(2) unless ``allow_overlap``."""
    eval_ranges = [[s, s + n] for s in eval_seeds]
    hyg: dict = {"eval_seed_ranges": eval_ranges, "allow_seed_overlap": allow_overlap}
    if train_config.get("data") != data:
        hyg.update({
            "checked": False, "source": None,
            "reason": "eval --data differs from train_config data; seed spaces are independent",
            "train_seed_range": None, "val_seed_range": None, "overlap_instances": 0,
        })
        return hyg
    seed = int(train_config.get("seed", 0))
    if "train_seed_range" in train_config:
        tr = [int(x) for x in train_config["train_seed_range"]]
        source = "train_config.train_seed_range"
    else:
        tr = [seed, seed + int(train_config.get("n_train", 0))]
        source = "reconstructed"
    if "val_seed_range" in train_config:
        vr = [int(x) for x in train_config["val_seed_range"]]
    else:
        vr = [seed + LEGACY_VAL_OFFSET, seed + LEGACY_VAL_OFFSET + LEGACY_VAL_SIZE]
    overlap, offending = 0, []
    for er in eval_ranges:
        for name, r in (("train", tr), ("val", vr)):
            lo, hi = max(er[0], r[0]), min(er[1], r[1])
            if lo < hi:
                overlap += hi - lo
                offending.append({"eval_range": er, "overlaps": name,
                                  "range": r, "n_instances": hi - lo})
    hyg.update({"checked": True, "source": source, "train_seed_range": tr,
                "val_seed_range": vr, "overlap_instances": overlap,
                "offending": offending})
    if overlap:
        msg = (f"SEED OVERLAP: eval instance-seed ranges intersect the model-selection "
               f"seed space: {offending} (train {tr}, val {vr})")
        if allow_overlap:
            print(f"WARNING: {msg} — proceeding under --allow-seed-overlap; "
                  f"results are contaminated by model selection.", file=sys.stderr)
        else:
            print(f"{msg} — refusing to eval on the model-selection set; "
                  f"pass --allow-seed-overlap to override.", file=sys.stderr)
            raise SystemExit(2)
    return hyg


def evaluate_full(policy, *, data: str, n: int, K: int, k_refine: int, T: int,
                  eval_seeds: list, families=None, ckpt: dict | None = None) -> dict:
    """W1 protocol: pooled multi-seed eval with enumeration baseline, timing,
    Holm-corrected comparisons, and no_context/policy_value arms.

    ``eval_seeds`` is the list of actual dataset seeds (already strided).
    ``ckpt`` (the loaded checkpoint dict) is needed only for the no_context arm.
    """
    checker = Checker()

    # P1 arm: context-ablated twin of the SAME weights (zero-param-change).
    nc_policy = None
    if ckpt is not None:
        nc_policy = StructurePolicy(**{**ckpt["model_kwargs"], "ablate_context": True})
        nc_policy.load_state_dict(ckpt["model_state_dict"])
        nc_policy.eval()

    rows = []
    per_seed = []
    for s in eval_seeds:
        instances = make_dataset(data, n, s, K=K, families=families)
        solver = load_solver(REFERENCE_SOLVER["name"], seed=s)
        rng = random.Random(s)
        gen = torch.Generator().manual_seed(s)
        gen_nc = torch.Generator().manual_seed(s)

        seed_rows = []
        for inst in instances:
            Ki = len(inst.candidates)
            cache: dict = {}

            def solved(choice, _inst=inst, _cache=cache, key=None, graph=None):
                """Best-of-k_refine refine + Checker against the chosen graph.
                ``key``/``graph`` override for non-menu graphs (policy_value)."""
                ck = key if key is not None else choice
                if ck not in _cache:
                    g = graph if graph is not None else (
                        _inst.fixed_graph if choice is None
                        else _inst.candidates[choice].apply(_inst.fixed_graph)
                    )
                    prob = Problem(id=f"{_inst.id}_arm{ck}", graph=g,
                                   solution=[0.0] * len(g.variables))
                    cands = [c for c in solver.sample(prob, k_refine) if c is not None]
                    _cache[ck] = checker.first_accepted(g, cands) is not None
                return _cache[ck]

            # 1) enumeration FIRST on the cold cache: honest wall-clock for the
            #    try-every-candidate baseline; later arms reuse the warm cache.
            t0 = time.perf_counter()
            order = random.Random(f"enum:{inst.id}:{s}").sample(range(Ki), Ki)
            accept, calls = None, 0
            for j in order:
                calls += 1
                if solved(j):
                    accept = j
                    break
            enum_row = {
                "solved": accept is not None,
                "accept_is_gold": accept == inst.gold_idx,
                "calls": calls,
                "wall_s": time.perf_counter() - t0,
            }

            # 2) control arms (warm cache).
            r = rng.randrange(Ki + 1)
            row = {
                "seed": s,
                "family": inst.family,
                "gold_idx": inst.gold_idx,
                "hard_idxs": sorted(_hard_negative_idxs(inst)),
                "certificate": getattr(inst, "certificate", "exact"),  # P3b
                "fixed_ok": solved(None),
                "gold_ok": solved(inst.gold_idx),
                "random_ok": solved(None if r == Ki else r),
                "enum": enum_row,
            }

            # 3) policy arms, timed per forward.
            rev_final = None
            for sampler, single in (("reverse", False), ("single_shot", True)):
                t0 = time.perf_counter()
                final, logits = reverse_sample(
                    policy, inst, T=T, generator=gen, single_shot=single
                )
                dt = time.perf_counter() - t0
                pick = chosen_candidate(final, logits, Ki)
                row[sampler] = {"pick": pick, "ok": solved(pick), "forward_s": dt}
                if sampler == "reverse":
                    rev_final = final

            # 4) no_context arm (P1), same seeds via a twin generator.
            if nc_policy is not None:
                nc = {}
                for sampler, single in (("reverse", False), ("single_shot", True)):
                    final, logits = reverse_sample(
                        nc_policy, inst, T=T, generator=gen_nc, single_shot=single
                    )
                    pick = chosen_candidate(final, logits, Ki)
                    nc[sampler] = {"pick": pick, "ok": solved(pick)}
                row["no_context"] = nc

            # 5) policy_value arm (P2/P3): does the predicted pin VALUE solve?
            pick = row["reverse"]["pick"]
            pv = {"solved_raw": False, "solved_snapped": False,
                  "fallback": False, "abs_err": None}
            if pick is not None:
                cand = inst.candidates[pick]
                pin = predicted_pin(rev_final, pick)
                if pin is None or getattr(cand, "defining_expression", None):
                    # P3: a scalar pin can't override a symbolic definition —
                    # silent fallback to the candidate's own pin semantics.
                    ok = solved(pick)
                    pv = {"solved_raw": ok, "solved_snapped": ok,
                          "fallback": True, "abs_err": None}
                else:
                    pin = float(pin)
                    snapped = float(round(pin))
                    g_raw = dataclasses.replace(cand, pin_value=pin).apply(inst.fixed_graph)
                    g_snap = dataclasses.replace(cand, pin_value=snapped).apply(inst.fixed_graph)
                    pv = {
                        "solved_raw": solved(None, key=("pv", pick, round(pin, 6)), graph=g_raw),
                        "solved_snapped": solved(None, key=("pv", pick, round(snapped, 6)), graph=g_snap),
                        "fallback": False,
                        "abs_err": abs(pin - cand.pin_value),
                    }
            row["pv"] = pv

            seed_rows.append(row)
        rows.extend(seed_rows)

        ns = len(seed_rows)
        per_seed.append({
            "seed": s,
            "n": ns,
            "fixed_k": sum(r["fixed_ok"] for r in seed_rows),
            "gold_k": sum(r["gold_ok"] for r in seed_rows),
            "random_k": sum(r["random_ok"] for r in seed_rows),
            "samplers": {
                sampler: {
                    "invention_k": sum(r[sampler]["pick"] == r["gold_idx"] for r in seed_rows),
                    "none_k": sum(r[sampler]["pick"] is None for r in seed_rows),
                    "solve_k": sum(r[sampler]["ok"] for r in seed_rows),
                } for sampler in ("reverse", "single_shot")
            },
            "enumeration": {
                "solve_k": sum(r["enum"]["solved"] for r in seed_rows),
                "first_accept_gold_k": sum(r["enum"]["accept_is_gold"] for r in seed_rows),
            },
        })

    # ---- pooled headline blocks --------------------------------------------
    N = len(rows)
    gold_k = sum(r["gold_ok"] for r in rows)
    fix_k = sum(r["fixed_ok"] for r in rows)
    rnd_k = sum(r["random_ok"] for r in rows)
    positive_ok = N > 0 and gold_k / N >= 0.95
    if not positive_ok:
        print(
            f"WARNING: POSITIVE CONTROL FAILED — gold_oracle solve rate "
            f"{gold_k}/{N} < 0.95. Solve-rate comparisons are not meaningful.",
            file=sys.stderr,
        )

    samplers = {}
    for sampler in ("reverse", "single_shot"):
        inv_k = sum(r[sampler]["pick"] == r["gold_idx"] for r in rows)
        none_k = sum(r[sampler]["pick"] is None for r in rows)
        misses = [r for r in rows if r[sampler]["pick"] != r["gold_idx"]]
        hn_k = sum(1 for r in misses
                   if r[sampler]["pick"] in r["hard_idxs"])
        pol_k = sum(r[sampler]["ok"] for r in rows)
        z_r, p_r = two_proportion_z(pol_k, N, rnd_k, N)
        z_f, p_f = two_proportion_z(pol_k, N, fix_k, N)
        per_family = {}
        for fam in sorted({r["family"] for r in rows}):
            fr = [r for r in rows if r["family"] == fam]
            per_family[fam] = {
                "invention_rate": rate_cell(
                    sum(r[sampler]["pick"] == r["gold_idx"] for r in fr), len(fr)),
                "solve_policy": rate_cell(sum(r[sampler]["ok"] for r in fr), len(fr)),
            }
        samplers[sampler] = {
            "invention_rate": rate_cell(inv_k, N),
            "none_rate": rate_cell(none_k, N),
            "hard_negative_confusion": rate_cell(hn_k, len(misses)),
            "solve": {
                "fixed": rate_cell(fix_k, N),
                "policy": rate_cell(pol_k, N),
                "random_slot": rate_cell(rnd_k, N),
                "always_none": rate_cell(fix_k, N),
                "gold_oracle": rate_cell(gold_k, N),
            },
            "comparisons": {
                "policy_vs_random": {"z": z_r, "p": p_r},
                "policy_vs_fixed": {"z": z_f, "p": p_f},
            },
            "per_family": per_family,
        }

    # ---- arms ---------------------------------------------------------------
    enum_k = sum(r["enum"]["solved"] for r in rows)
    gold_first_k = sum(r["enum"]["accept_is_gold"] for r in rows)
    enum_walls = [r["enum"]["wall_s"] for r in rows]
    calls_mean = (sum(r["enum"]["calls"] for r in rows) / N) if N else 0.0
    arms: dict = {
        "enumeration": {
            "status": "ok",
            "solve": rate_cell(enum_k, N),
            "first_accept_is_gold": rate_cell(gold_first_k, enum_k),
            "solved_calls_mean": calls_mean,
            "refine_calls_mean": calls_mean * k_refine,
            "wall_clock_s": {"mean": (sum(enum_walls) / N) if N else 0.0,
                             "total": sum(enum_walls)},
        }
    }
    if nc_policy is None:
        arms["no_context"] = {"status": "skipped",
                              "reason": "no checkpoint provided (stub/test run)"}
    else:
        arms["no_context"] = {
            "status": "ok",
            "samplers": {
                sampler: {
                    "invention_rate": rate_cell(
                        sum(r["no_context"][sampler]["pick"] == r["gold_idx"] for r in rows), N),
                    "solve_policy": rate_cell(
                        sum(r["no_context"][sampler]["ok"] for r in rows), N),
                } for sampler in ("reverse", "single_shot")
            },
        }
    errs = [r["pv"]["abs_err"] for r in rows if r["pv"]["abs_err"] is not None]
    arms["policy_value"] = {
        "status": "ok",
        "solve_raw": rate_cell(sum(r["pv"]["solved_raw"] for r in rows), N),
        "solve_snapped": rate_cell(sum(r["pv"]["solved_snapped"] for r in rows), N),
        "pin_abs_err_mean": (sum(errs) / len(errs)) if errs else None,
        "value_fallback": sum(r["pv"]["fallback"] for r in rows),
    }

    # ---- Holm-corrected declared comparison family --------------------------
    tests: dict = {}
    for sampler in ("reverse", "single_shot"):
        pol_k = sum(r[sampler]["ok"] for r in rows)
        z, p = two_proportion_z(pol_k, N, rnd_k, N)
        tests[f"{sampler}:policy_vs_random"] = {"z": z, "p": p}
        z, p = two_proportion_z(pol_k, N, fix_k, N)
        tests[f"{sampler}:policy_vs_fixed"] = {"z": z, "p": p}
    if nc_policy is not None:
        for sampler in ("reverse", "single_shot"):
            pol_k = sum(r[sampler]["ok"] for r in rows)
            nc_k = sum(r["no_context"][sampler]["ok"] for r in rows)
            z, p = two_proportion_z(pol_k, N, nc_k, N)
            tests[f"{sampler}:policy_vs_no_context"] = {"z": z, "p": p}
    pv_k = sum(r["pv"]["solved_raw"] for r in rows)
    z, p = two_proportion_z(pv_k, N, rnd_k, N)
    tests["reverse:policy_value_vs_random"] = {"z": z, "p": p}
    names = list(tests)
    for name, ph in zip(names, _holm([tests[nm]["p"] for nm in names])):
        tests[name]["p_holm"] = ph
        tests[name]["significant_05"] = ph < 0.05
    comparisons_holm = {"method": "holm", "alpha": 0.05, "m": len(tests), "tests": tests}

    # ---- timing / amortization ----------------------------------------------
    fwd = [r[sampler]["forward_s"] for r in rows for sampler in ("reverse", "single_shot")]
    pf_mean = (sum(fwd) / len(fwd)) if fwd else 0.0
    en_mean = arms["enumeration"]["wall_clock_s"]["mean"]
    timing = {
        "policy_forward_s_mean": pf_mean,
        "enumeration_s_mean": en_mean,
        "amortization_note": (
            f"enumeration solves the certified linear menu exactly "
            f"(solve rate {arms['enumeration']['solve']['rate']:.2f}) in "
            f"{en_mean:.4f}s of solver calls per instance; one policy forward costs "
            f"{pf_mean:.4f}s. The policy's contribution is amortizing solver calls "
            f"into a learned forward pass, not exceeding enumeration's capability."
        ),
    }

    certificates: dict = {}
    for r in rows:
        certificates[r["certificate"]] = certificates.get(r["certificate"], 0) + 1
    positive_control = {
        "gold_oracle": rate_cell(gold_k, N),
        "threshold": 0.95,
        "ok": positive_ok,
        "by_construction": [
            "generator resamples lambda_max(A^T A)>=9",
            "polish_steps=4000 (default 400)",
        ],
        "certificates": certificates,
    }

    caveats = [
        "single training seed; CIs cover instance variance only",
        "headline rates pool counts across eval seeds; per-seed counts in per_seed",
    ]

    return {
        "positive_control_ok": positive_ok,
        "samplers": samplers,
        "arms": arms,
        "comparisons_holm": comparisons_holm,
        "timing": timing,
        "positive_control": positive_control,
        "per_seed": per_seed,
        "caveats": caveats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W1 end-to-end invention eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="instances per eval seed")
    ap.add_argument("--k-refine", type=int, default=4)
    ap.add_argument("--data", choices=SOURCES, default="toys")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--eval-seeds", type=int, default=5, help="number of eval seeds")
    ap.add_argument("--seed-base", type=int, default=TEST_SEED_BASE)
    ap.add_argument("--families", default=None,
                    help="comma-separated family subset (default: all)")
    ap.add_argument("--allow-seed-overlap", action="store_true",
                    help="proceed despite eval/train-val seed overlap (loud warning)")
    args = ap.parse_args()

    if args.n > SEED_STRIDE:
        ap.error(f"--n {args.n} > SEED_STRIDE {SEED_STRIDE}: per-seed instance "
                 f"ranges would collide across eval seeds")
    if args.n < 1 or args.eval_seeds < 1:
        ap.error("--n and --eval-seeds must be >= 1")

    families = None
    if args.families:
        allowed = tuple(getattr(invention_data, "FAMILIES_BY_SOURCE", {})
                        .get(args.data, FAMILIES))  # P4
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        bad = [f for f in families if f not in allowed]
        if bad:
            ap.error(f"--families {bad} not in {allowed} for --data {args.data}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        policy = StructurePolicy(**ckpt["model_kwargs"])
        policy.load_state_dict(ckpt["model_state_dict"])
    except Exception as exc:  # missing/unloadable ckpt is an expected skip, not a crash
        report = {
            "status": "skipped",
            "reason": f"checkpoint {args.ckpt!r} missing or unloadable: {exc}",
        }
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"skipped: {report['reason']}")
        return 0

    policy.eval()
    train_config = ckpt.get("train_config", {})
    T = int(train_config.get("T", 20))
    eval_seeds = [args.seed_base + j * SEED_STRIDE for j in range(args.eval_seeds)]

    seed_hygiene = _check_seed_hygiene(
        train_config, args.data, eval_seeds, args.n, args.allow_seed_overlap
    )

    # P4: data-module version vs the version the checkpoint was trained on.
    ev = int(getattr(invention_data, "DATA_VERSION", 1))
    cv = int(train_config.get("data_version", 1))
    if ev != cv:
        print(f"WARNING: DATA_VERSION mismatch — eval data module v{ev}, "
              f"checkpoint trained on v{cv}.", file=sys.stderr)

    res = evaluate_full(
        policy, data=args.data, n=args.n, K=args.K, k_refine=args.k_refine,
        T=T, eval_seeds=eval_seeds, families=families, ckpt=ckpt,
    )
    report = {
        "status": "ok",
        "config": {
            "ckpt": args.ckpt,
            "n": args.n,
            "k_refine": args.k_refine,
            "data": args.data,
            "K": args.K,
            "T": T,
            "eval_seeds": eval_seeds,
            "seed_base": args.seed_base,
            "families": families,
            "solver": REFERENCE_SOLVER,
            "data_version": {"eval": ev, "ckpt": cv, "match": ev == cv},
        },
        "seed_hygiene": seed_hygiene,
        **res,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    for sampler in ("reverse", "single_shot"):
        b = report["samplers"][sampler]
        print(
            f"[{sampler}] invention {b['invention_rate']['rate']:.3f}  "
            f"solve: policy {b['solve']['policy']['rate']:.3f}  "
            f"fixed {b['solve']['fixed']['rate']:.3f}  "
            f"random {b['solve']['random_slot']['rate']:.3f}  "
            f"gold {b['solve']['gold_oracle']['rate']:.3f}"
        )
    e = report["arms"]["enumeration"]
    print(f"[enumeration] solve {e['solve']['rate']:.3f}  "
          f"first_accept_is_gold {e['first_accept_is_gold']['rate']:.3f}  "
          f"wall {e['wall_clock_s']['mean']:.4f}s/inst")
    print(f"positive_control_ok={report['positive_control_ok']}  "
          f"seed_overlap={report['seed_hygiene']['overlap_instances']}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
