#!/usr/bin/env python3
"""Cheap-probe baseline for the structural repair ranker (R10 reviewer control).

The obvious attack on a learned repair ranker: why *learn* which candidate repair
to try when you could probe each of the K candidates with a tiny solver budget and
keep the best?  This script runs that control.  Per instance:

  1. PROBE: each candidate augmentation gets a short-budget solve (``--restarts``
     random starts through ``marc.refine.iterative.refine`` at a small
     steps/polish budget; several budget settings via ``--budgets``).  The pick is
     the first candidate the checker accepts at probe budget, else the candidate
     with the lowest best-seen energy (residual).
  2. GRADE: the pick is then graded at the full reference budget through the
     *identical* ``_solves`` path the other arms use (common random numbers:
     ``solve_seed = eval_seed + 100003 * i``), so probe quality is measured on
     exactly the same footing as full ranker / candidate-only / random /
     enumeration.

Everything structural is imported from ``run_repair_ranker`` (data building,
``_solves``, the {k, n, rate, ci95} cell format) — that script is NOT modified.
The ranker arms are optional: pass ``--ckpt`` (a run_repair_ranker checkpoint) to
score full/candidate-only on the same instances; without it those arms are
skipped with a note (protocol still identical for the rest).

Costs reported per arm: invention accuracy, e2e solve rate, mean solver calls
(probe short-calls and full-budget grading calls counted separately), wall-clock.
Menus at K = 4, 8, 16 (``--Ks``), same seed protocol as run_repair_ranker
(test split at ``seed + 900000``, eval stream at ``seed + 42``).

Interesting either way: if the trained ranker beats the probe at matched
wall-clock, that is a strong new paper row; if the probe matches the ranker,
R10 needs reframing before submission.

Writes results/p_repair/probe_baseline.json.

Run (pilot):  PYTHONPATH=. python3 scripts/run_probe_baseline.py --quick
Full:         PYTHONPATH=. python3 scripts/run_probe_baseline.py \
                  --ckpt checkpoints/repair_ranker.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.cas.checker import Checker
from marc.eval.metrics import two_proportion_z
from marc.refine.iterative import refine
from marc.structure.invention_data import DATA_VERSION, FAMILIES_BY_SOURCE

# --- import the ranker experiment module (data building, _solves, cell format) ---
_spec = importlib.util.spec_from_file_location(
    "run_repair_ranker", Path(__file__).resolve().parent / "run_repair_ranker.py"
)
rrr = importlib.util.module_from_spec(_spec)
sys.modules["run_repair_ranker"] = rrr  # dataclass resolution needs the module registered
_spec.loader.exec_module(rrr)

build_split = rrr.build_split
_solves = rrr._solves          # full reference-budget grading (per source)
_rate = rrr._rate              # {k, n, rate, ci95} cell format
_batch = rrr._batch


def parse_budgets(spec: str) -> list[dict]:
    """"60:100,150:200" -> [{"steps": 60, "polish_steps": 100}, ...]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        steps, polish = part.split(":")
        out.append({"steps": int(steps), "polish_steps": int(polish)})
    if not out:
        raise ValueError(f"no budgets parsed from {spec!r}")
    return out


def probe_pick(pack, budget: dict, restarts: int, seed: int,
               checker: Checker) -> tuple[int, int, float]:
    """Short-budget probe over the K candidates.

    Returns (pick, probe_calls, wall_s).  Pick = first candidate the checker
    accepts at probe budget (scan order = candidate order, matching enumeration's
    per-candidate cost model), else argmin of the best energy seen.  ``noise=False``
    keeps the probe deterministic per (instance, restart seed).
    """
    t0 = time.perf_counter()
    K = len(pack.inst.candidates)
    best_j, best_e = 0, float("inf")
    calls = 0
    for j in range(K):
        graph = pack.inst.candidates[j].apply(pack.inst.fixed_graph)
        nv = len(graph.variables)
        e_j = float("inf")
        accepted = False
        for r in range(restarts):
            rng = np.random.default_rng(seed + 7919 * j + 104729 * r)
            x0 = (rng.standard_normal(nv) * 3.0).tolist()  # matches solver init_scale
            calls += 1
            trace = refine(graph, x0, noise=False, seed=seed + r, **budget)
            e_j = min(e_j, float(trace.best_energy))
            if checker.first_accepted(graph, [trace.x]) is not None:
                accepted = True
                break
        if accepted:
            return j, calls, time.perf_counter() - t0
        if e_j < best_e:
            best_j, best_e = j, e_j
    return best_j, calls, time.perf_counter() - t0


@torch.no_grad()
def ranker_picks(ckpt_path: str, packs, batch_size: int) -> dict[str, list[int]] | None:
    """Optional full/candidate-only arms from a run_repair_ranker checkpoint."""
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    payload = torch.load(ckpt_path, map_location="cpu")
    kw = payload["model_kwargs"]
    full = rrr.GraphRepairRanker(D=int(kw["D"]), L=int(kw["L"]))
    control = rrr.CandidateOnlyRanker(D=int(kw["D"]))
    full.load_state_dict(payload["full_state_dict"])
    control.load_state_dict(payload["control_state_dict"])
    full.eval()
    control.eval()
    picks: dict[str, list[int]] = {"full": [], "control": []}
    device = torch.device("cpu")
    for start in range(0, len(packs), batch_size):
        ps = packs[start:start + batch_size]
        gb, cf, _labels, K = _batch(ps, device)
        picks["full"] += full(gb).reshape(len(ps), K).argmax(1).tolist()
        picks["control"] += control(cf).reshape(len(ps), K).argmax(1).tolist()
    return picks


def evaluate_K(K: int, args, checker: Checker) -> dict:
    """All arms on one K-sized menu suite; same split/seed protocol as run_repair_ranker."""
    sources = [s.strip() for s in args.eval_data.split(",") if s.strip()]
    # Build per source so an infeasible (source, K) combo skips instead of aborting:
    # the nonlinear generator cannot fill exchangeable menus beyond K=4 (same reason
    # the run_repair_ranker K=8/16 result files are linear-only).
    packs = []
    skipped = {}
    for source in sources:
        try:
            packs += build_split([source], args.n_test, args.seed + 900000, K)
        except RuntimeError as exc:
            skipped[source] = str(exc)
            print(f"  [skip] {source} at K={K}: {exc}", flush=True)
    if not packs:
        return {"K": K, "n": 0, "skipped_sources": skipped, "arms": {}}
    n = len(packs)
    eval_seed = args.seed + 42
    rng = random.Random(eval_seed)
    budgets = parse_budgets(args.budgets)

    # picks + probe cost per arm
    arm_picks: dict[str, list[int]] = {f"probe[{b['steps']}:{b['polish_steps']}]": []
                                       for b in budgets}
    probe_cost = {a: {"calls": 0, "wall_s": 0.0} for a in arm_picks}
    arm_picks["random"] = [rng.randrange(K) for _ in range(n)]

    for i, pack in enumerate(packs):
        for b in budgets:
            arm = f"probe[{b['steps']}:{b['polish_steps']}]"
            pick, calls, wall = probe_pick(
                pack, b, args.restarts, eval_seed + 100003 * i, checker
            )
            arm_picks[arm].append(pick)
            probe_cost[arm]["calls"] += calls
            probe_cost[arm]["wall_s"] += wall

    rk = ranker_picks(args.ckpt, packs, args.batch_size)
    if rk is not None:
        arm_picks["full"] = rk["full"]
        arm_picks["control"] = rk["control"]

    result: dict = {"K": K, "n": n, "arms": {}}
    if skipped:
        result["skipped_sources"] = skipped

    # invention accuracy (pick == gold) for every arm
    for arm, picks in arm_picks.items():
        k_inv = sum(p == pack.inst.gold_idx for p, pack in zip(picks, packs))
        result["arms"][arm] = {"invention": _rate(k_inv, n)}

    # e2e solve at the full reference budget — identical _solves + common random
    # numbers across arms (solve_seed depends only on the instance index)
    grade_cost = {arm: {"full_calls": 0, "wall_s": 0.0} for arm in arm_picks}
    solved = {arm: 0 for arm in arm_picks}
    solved["oracle"] = 0
    solved["enumeration"] = 0
    enum_calls = 0
    enum_wall = 0.0
    for i, pack in enumerate(packs):
        solve_seed = eval_seed + 100003 * i
        for arm, picks in arm_picks.items():
            t0 = time.perf_counter()
            solved[arm] += int(_solves(pack, picks[i], solve_seed))
            grade_cost[arm]["full_calls"] += 1
            grade_cost[arm]["wall_s"] += time.perf_counter() - t0
        solved["oracle"] += int(_solves(pack, pack.inst.gold_idx, solve_seed))
        order = list(range(K))
        random.Random(solve_seed + 11).shuffle(order)
        for pick in order:
            enum_calls += 1
            t0 = time.perf_counter()
            ok = _solves(pack, pick, solve_seed)
            enum_wall += time.perf_counter() - t0
            if ok:
                solved["enumeration"] += 1
                break

    for arm in arm_picks:
        cell = result["arms"][arm]
        cell["solve"] = _rate(solved[arm], n)
        calls = probe_cost.get(arm, {}).get("calls", 0)
        wall = probe_cost.get(arm, {}).get("wall_s", 0.0)
        cell["cost"] = {
            "probe_calls_per_instance": calls / n,
            "grade_calls_per_instance": grade_cost[arm]["full_calls"] / n,
            "solver_calls_per_instance": (calls + grade_cost[arm]["full_calls"]) / n,
            "wall_s_total": wall + grade_cost[arm]["wall_s"],
            "wall_s_per_instance": (wall + grade_cost[arm]["wall_s"]) / n,
        }
    result["arms"]["oracle"] = {"solve": _rate(solved["oracle"], n)}
    result["arms"]["enumeration"] = {
        "solve": _rate(solved["enumeration"], n),
        "cost": {
            "solver_calls_per_instance": enum_calls / n,
            "wall_s_total": enum_wall,
            "wall_s_per_instance": enum_wall / n,
        },
    }

    # head-to-heads: ranker vs each probe budget (the reviewer question), on both
    # invention and solve; one-sided z, ranker > probe
    if rk is not None:
        result["full_vs_probe"] = {}
        kf = result["arms"]["full"]["invention"]["k"]
        for b in budgets:
            arm = f"probe[{b['steps']}:{b['polish_steps']}]"
            kp = result["arms"][arm]["invention"]["k"]
            z, p = two_proportion_z(kf, n, kp, n)
            zs, ps = two_proportion_z(
                result["arms"]["full"]["solve"]["k"], n,
                result["arms"][arm]["solve"]["k"], n,
            )
            result["full_vs_probe"][arm] = {
                "invention": {"z": z, "p_one_sided": p},
                "solve": {"z": zs, "p_one_sided": ps},
            }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="cheap-probe baseline for the repair ranker")
    ap.add_argument("--eval-data", default="aux_required,nonlinear",
                    help="comma list of sources (matches run_repair_ranker --eval-data)")
    ap.add_argument("--n-test", type=int, default=500, help="instances per source")
    ap.add_argument("--Ks", default="4,8,16", help="menu sizes")
    ap.add_argument("--budgets", default="60:100,150:200,300:400",
                    help="probe budgets steps:polish_steps, comma-separated")
    ap.add_argument("--restarts", type=int, default=1, help="probe restarts per candidate")
    ap.add_argument("--seed", type=int, default=20260722,
                    help="MUST match the run_repair_ranker run being compared against")
    ap.add_argument("--ckpt", default="checkpoints/repair_ranker.pt",
                    help="optional run_repair_ranker checkpoint for the ranker arms")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/p_repair/probe_baseline.json")
    ap.add_argument("--quick", action="store_true", help="pilot scale")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_test = 48
        args.Ks = "4,8"

    bad = [s for s in args.eval_data.split(",") if s.strip() not in FAMILIES_BY_SOURCE]
    if bad:
        ap.error(f"unknown data sources: {bad}")

    checker = Checker()
    t0 = time.time()
    by_K = {}
    for K in [int(k) for k in args.Ks.split(",")]:
        print(f"== K={K} ==", flush=True)
        by_K[str(K)] = evaluate_K(K, args, checker)
        r = by_K[str(K)]
        for arm, cell in r["arms"].items():
            inv = cell.get("invention", {}).get("rate")
            sv = cell.get("solve", {}).get("rate")
            calls = cell.get("cost", {}).get("solver_calls_per_instance")
            print(f"  {arm:>22}: invention={inv if inv is None else f'{inv:.3f}'}"
                  f"  solve={sv if sv is None else f'{sv:.3f}'}"
                  f"  calls/inst={calls if calls is None else f'{calls:.2f}'}",
                  flush=True)

    payload = {
        "status": "ok",
        "method": "cheap-probe candidate selection (short-budget solve per candidate, "
                  "pick first-accept else lowest residual), graded at full reference budget",
        "data_version": DATA_VERSION,
        "config": vars(args),
        "ranker_arms": "included" if Path(args.ckpt).exists() else
                       f"skipped (no checkpoint at {args.ckpt}); rerun with --ckpt "
                       "after the run_repair_ranker training run",
        "seed_hygiene": {
            "test": [args.seed + 900000, args.seed + 900000 + args.n_test],
            "note": "test-only script; split seed formula identical to run_repair_ranker",
            "overlap_instances": 0,
        },
        "by_K": by_K,
        "wall_s": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}; wall={payload['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
