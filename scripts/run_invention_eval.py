"""End-to-end invention eval (U5): does the trained policy's picked augmentation
actually make the graph solvable?

Five arms per test instance (seeds disjoint from training), each solved
best-of-``k_refine`` with the "refine" solver and verified by the Checker against
the graph actually solved:

    fixed        — the unaugmented (inconsistent) graph: the H2 baseline
    policy       — reverse_sample -> chosen_candidate -> Candidate.apply
                   (a None pick falls back to the fixed graph)
    random_slot  — uniform over the K candidates + "none" (seeded)
    always_none  — always the fixed graph
    gold_oracle  — the gold candidate (positive control: must solve >= 0.95)

Both samplers are reported: "reverse" (iterative D3PM ancestral) and
"single_shot" (one forward at t=T-1, the ablation). Every rate is
{k, n, rate, ci95} via wilson_interval; comparisons use two_proportion_z.

Missing/unloadable checkpoint -> writes {"status": "skipped", ...} and exits 0.

Usage (CLI contract C4):
    python3 scripts/run_invention_eval.py --ckpt checkpoints/structure_policy.pt \
        --out results/p5_invention/invention.json [--n 100] [--k-refine 4] \
        [--data toys|aux_required] [--seed 500000] [--K 4]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marc.cas.checker import Checker
from marc.eval.metrics import two_proportion_z, wilson_interval
from marc.eval.runner import Problem
from marc.eval.solver import load_solver
from marc.structure.invention_data import InventionInstance, make_dataset
from marc.structure.policy import StructurePolicy, chosen_candidate, reverse_sample


def _rate(k: int, n: int) -> dict:
    """{k, n, rate, ci95} block; degenerate n=0 (e.g. zero misses) -> vacuous CI."""
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci95": [0.0, 1.0]}
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def _hard_negative_idx(inst: InventionInstance) -> int | None:
    """Index of the gold-structure/wrong-value distractor, if the menu has one."""
    gold = inst.candidates[inst.gold_idx]
    for j, c in enumerate(inst.candidates):
        if j != inst.gold_idx and c.insert_coeffs == gold.insert_coeffs:
            return j
    return None


def evaluate(policy, instances, *, k_refine: int = 4, T: int = 20, seed: int = 0) -> dict:
    """Run all arms over ``instances``; returns {"positive_control_ok", "samplers"}."""
    # ponytail: polish_steps=4000 (default 400) — the checker's symbolic snap gate
    # needs ~1e-10 accuracy; slow eigenmodes (lambda_min ~ 0.2 on some augmented
    # systems) need the longer noise-free tail to get there. Pure config on the
    # frozen solver; drop if refine ever gets adaptive lr.
    solver = load_solver("refine", seed=seed, polish_steps=4000)
    checker = Checker()
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    rows = []
    for inst in instances:
        K = len(inst.candidates)
        cache: dict = {}

        def solved(choice, _inst=inst, _cache=cache):
            """Best-of-k_refine refine + Checker.verify against the chosen graph."""
            if choice not in _cache:
                graph = (
                    _inst.fixed_graph if choice is None
                    else _inst.candidates[choice].apply(_inst.fixed_graph)
                )
                prob = Problem(
                    id=f"{_inst.id}_arm{choice}",
                    graph=graph,
                    solution=[0.0] * len(graph.variables),  # unused by refine
                )
                cands = [c for c in solver.sample(prob, k_refine) if c is not None]
                _cache[choice] = checker.first_accepted(graph, cands) is not None
            return _cache[choice]

        r = rng.randrange(K + 1)
        row = {
            "family": inst.family,
            "gold_idx": inst.gold_idx,
            "hard_idx": _hard_negative_idx(inst),
            "fixed_ok": solved(None),
            "gold_ok": solved(inst.gold_idx),
            "random_ok": solved(None if r == K else r),
        }
        for sampler, single in (("reverse", False), ("single_shot", True)):
            final, logits = reverse_sample(
                policy, inst, T=T, generator=gen, single_shot=single
            )
            pick = chosen_candidate(final, logits, K)
            row[sampler] = {"pick": pick, "ok": solved(pick)}
        rows.append(row)

    n = len(rows)
    gold_k = sum(r["gold_ok"] for r in rows)
    positive_ok = n > 0 and gold_k / n >= 0.95
    if not positive_ok:
        print(
            f"WARNING: POSITIVE CONTROL FAILED — gold_oracle solve rate "
            f"{gold_k}/{n} < 0.95. Solve-rate comparisons are not meaningful.",
            file=sys.stderr,
        )

    fix_k = sum(r["fixed_ok"] for r in rows)
    rnd_k = sum(r["random_ok"] for r in rows)
    samplers = {}
    for sampler in ("reverse", "single_shot"):
        inv_k = sum(r[sampler]["pick"] == r["gold_idx"] for r in rows)
        none_k = sum(r[sampler]["pick"] is None for r in rows)
        misses = [r for r in rows if r[sampler]["pick"] != r["gold_idx"]]
        hn_k = sum(
            1 for r in misses
            if r["hard_idx"] is not None and r[sampler]["pick"] == r["hard_idx"]
        )
        pol_k = sum(r[sampler]["ok"] for r in rows)
        z_r, p_r = two_proportion_z(pol_k, n, rnd_k, n)
        z_f, p_f = two_proportion_z(pol_k, n, fix_k, n)
        per_family = {}
        for fam in sorted({r["family"] for r in rows}):
            fr = [r for r in rows if r["family"] == fam]
            per_family[fam] = {
                "invention_rate": _rate(
                    sum(r[sampler]["pick"] == r["gold_idx"] for r in fr), len(fr)
                ),
                "solve_policy": _rate(sum(r[sampler]["ok"] for r in fr), len(fr)),
            }
        samplers[sampler] = {
            "invention_rate": _rate(inv_k, n),
            "none_rate": _rate(none_k, n),
            "hard_negative_confusion": _rate(hn_k, len(misses)),
            "solve": {
                "fixed": _rate(fix_k, n),
                "policy": _rate(pol_k, n),
                "random_slot": _rate(rnd_k, n),
                "always_none": _rate(fix_k, n),  # identical arm to fixed by definition
                "gold_oracle": _rate(gold_k, n),
            },
            "comparisons": {
                "policy_vs_random": {"z": z_r, "p": p_r},
                "policy_vs_fixed": {"z": z_f, "p": p_f},
            },
            "per_family": per_family,
        }
    return {"positive_control_ok": positive_ok, "samplers": samplers}


def main() -> int:
    ap = argparse.ArgumentParser(description="U5 end-to-end invention eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k-refine", type=int, default=4)
    ap.add_argument("--data", choices=("toys", "aux_required"), default="toys")
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()

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
    T = int(ckpt.get("train_config", {}).get("T", 20))
    instances = make_dataset(args.data, args.n, args.seed, K=args.K)
    res = evaluate(policy, instances, k_refine=args.k_refine, T=T, seed=args.seed)
    report = {
        "status": "ok",
        "config": {
            "ckpt": args.ckpt,
            "n": args.n,
            "k_refine": args.k_refine,
            "data": args.data,
            "seed": args.seed,
            "K": args.K,
            "T": T,
        },
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
    print(f"positive_control_ok={report['positive_control_ok']}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
