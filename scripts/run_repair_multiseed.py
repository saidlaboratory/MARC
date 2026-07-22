#!/usr/bin/env python3
"""Optimization-seed robustness for the structural repair ranker.

Certified train/validation/test menus are generated once and held fixed; only
parameter initialization and training order change.  This cleanly measures model
optimization variance without conflating it with instance sampling variance (which
is already reported by Wilson intervals on the common test set).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_repair_ranker as rr
from marc.model.repair_ranker import CandidateOnlyRanker, GraphRepairRanker
from marc.structure.invention_data import DATA_VERSION


def _run_one_seed(seed: int, args, train_set, val_set, test_set, n_workers: int) -> dict:
    """Train + evaluate one optimization seed on the shared splits.

    Module-level so it pickles for the --jobs fan-out; seeds are independent
    after build_split, so this is embarrassingly parallel and CPU-bound.
    The eval seed VARIES with the optimization seed (issue #103): the test
    instances stay fixed, but each run's stochastic arm draws (the random
    baseline) must be independent — the old constant ``data_seed + 42`` made
    the random arm byte-identical across seeds, so ``population_sd`` read 0.0
    from a single draw copied three times and every full-vs-random delta
    reused one correlated draw.
    """
    if n_workers > 1:  # avoid thread oversubscription when fanned out
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // n_workers))
    device = torch.device(args.device)
    print(f"=== optimization seed {seed} ===", flush=True)
    torch.manual_seed(seed)
    full = GraphRepairRanker(D=args.D, L=args.L)
    control = CandidateOnlyRanker(D=args.D)
    history, best_f, best_c = rr.train(
        full, control, train_set, val_set, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=seed, device=device,
    )
    eval_seed = args.data_seed + 42 + 1_000_003 * seed
    result = rr.evaluate(
        full, control, test_set, batch_size=args.batch_size,
        device=device, seed=eval_seed, solve_e2e=False,
    )
    ckpt = Path(args.ckpt_dir) / f"seed_{seed}.pt"
    torch.save({
        "full_state_dict": {k: v.detach().cpu() for k, v in full.state_dict().items()},
        "control_state_dict": {k: v.detach().cpu() for k, v in control.state_dict().items()},
        "model_kwargs": {"D": args.D, "L": args.L},
        "train_seed": seed,
        "data_seed": args.data_seed,
    }, ckpt)
    row = {
        "train_seed": seed,
        "eval_seed": eval_seed,
        "best_validation": {"full": best_f, "control": best_c},
        "full": result["full"]["invention"],
        "control": result["control"]["invention"],
        "random": result["random"]["invention"],
        "paired": {
            "full_gt_control": result["full_gt_control"]["paired_mcnemar"],
            "full_gt_random": result["full_gt_random"]["paired_mcnemar"],
        },
        "per_family": result["per_family"],
        "checkpoint": str(ckpt),
        "last_history": history[-1],
    }
    print(f"seed {seed}: full={row['full']['rate']:.3f} "
          f"control={row['control']['rate']:.3f} random={row['random']['rate']:.3f}",
          flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="multi-seed repair-ranker robustness")
    ap.add_argument("--data", default="aux_required", choices=("aux_required", "nonlinear"))
    ap.add_argument("--exclude-family", action="append", default=[])
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--D", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--data-seed", type=int, default=20260722)
    ap.add_argument("--train-seeds", default="11,29,47")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/p_repair/multiseed.json")
    ap.add_argument("--ckpt-dir", default="checkpoints/repair_multiseed")
    ap.add_argument("--jobs", type=int, default=1,
                    help="fan the seeds out over N processes (independent after "
                         "build_split; CPU-bound, so N=len(seeds) ~ 1-seed wall time)")
    args = ap.parse_args(argv)
    seeds = [int(x) for x in args.train_seeds.split(",") if x.strip()]
    excluded = set(args.exclude_family)
    t0 = time.time()
    print("building one shared certified split", flush=True)
    train_set = rr.build_split([args.data], args.n_train, args.data_seed, args.K, excluded)
    val_set = rr.build_split(
        [args.data], args.n_val, args.data_seed + 500000, args.K, excluded
    )
    test_set = rr.build_split([args.data], args.n_test, args.data_seed + 900000, args.K)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    n_jobs = max(1, min(args.jobs, len(seeds)))
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_run_one_seed, s, args, train_set, val_set,
                                   test_set, n_jobs) for s in seeds]
            runs = [f.result() for f in futures]      # keep seed order
    else:
        runs = [_run_one_seed(s, args, train_set, val_set, test_set, 1)
                for s in seeds]

    def aggregate(arm):
        values = [r[arm]["rate"] for r in runs]
        return {
            "mean": statistics.mean(values),
            "population_sd": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    payload = {
        "status": "ok",
        "data_version": DATA_VERSION,
        "protocol": (
            "shared certified instances; independent initialization and training-order "
            "seeds; per-seed eval seed (data_seed + 42 + 1000003*train_seed) so "
            "stochastic arms draw independently across runs (issue #103); Wilson CIs "
            "within run cover instance variance"
        ),
        "config": vars(args),
        "runs": runs,
        "aggregate": {arm: aggregate(arm) for arm in ("full", "control", "random")},
        "wall_s": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}; full={payload['aggregate']['full']}", flush=True)


if __name__ == "__main__":
    main()
