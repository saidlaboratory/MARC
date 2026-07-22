"""Does a TRAINED denoiser beat random restart on the geometry point-chain family?
(The factorization law's live prediction, confirmed or refuted.)

The crossover law (scripts/run_crossover_theory.py) measured a steep reachability
collapse on ``make_point_chain`` (log q slope ~ -0.77, R^2 0.999) and flagged it as a
real-domain regime where an amortized proposal *should* help. That is a prediction
about a learned model we had not yet trained. This script trains one and checks.

Methodology is identical to the R5 dimension-scaling result the law generalizes from
(scripts/run_dimension_scaling.py): per chain length k we train a small GraphDenoiser
with an x0 objective on fresh point-chain instances, then evaluate learned vs. the
random-multistart control and Langevin, best-of-K, on held-out instances. Every arm
shares the geometry-tuned polish (marc/refine/presets, issue #104) and the exact
Checker gate after a 6-decimal snap, exactly as the crossover measurement did, so the
learned arm is compared on the same footing as the random control. The learned
proposal is the one-shot x0 readout net(data, T) (R5's inference path), scaled and
polished. Wilson CIs on every rate; a two-proportion z on learned>random per k.

Prediction (from the law): learned should hold where random collapses at large k
(k=3,4 -> n=6,8). If it does, the law made a prediction on a real-ish domain and it
held; if learned also collapses, the law's reachability measurement did not transfer
to what this denoiser can learn, and we say so.

Outputs results/p_geometry/pointchain_learned.json.
Run:  PYTHONPATH=. python3 scripts/run_pointchain_learned.py [--quick] [--trials 40] [--K 8]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from marc.cas.checker import Checker
from marc.data.geometry import make_point_chain
from marc.diffusion.forward import corrupt
from marc.diffusion.schedule import cosine_beta_schedule
from marc.eval.metrics import two_proportion_z, wilson_interval
from marc.graph.pyg import build_heterodata
from marc.model.denoiser import GraphDenoiser
from marc.refine.iterative import refine
from marc.refine.presets import GEOMETRY_INIT_SD as INIT_SD
from marc.refine.presets import GEOMETRY_POLISH_KWARGS as POLISH

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 4.0           # point-chain roots are small nonzero integers in [-4,4]
DECIMALS = 6          # snap-to-grid before the checker's exact gate
LANGEVIN = dict(POLISH, noise=True, sigma0=0.5)
KS = [1, 2, 3, 4]     # points per chain; n = 2k variables


def suite(k, count, seed0):
    return [make_point_chain(k, random.Random(seed0 + 7919 * j)) for j in range(count)]


def accepted(chk, g, x):
    return chk.verify(g, [round(v, DECIMALS) for v in x]).accepted


def train_x0(items, epochs, D=128, L=4, seed=0):
    """Small GraphDenoiser, x0 objective — identical to run_dimension_scaling."""
    torch.manual_seed(seed)
    net = GraphDenoiser(D=D, L=L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), torch.tensor([[v] for v in sol], dtype=torch.float32) / SCALE)
             for g, sol in items]
    for _ in range(epochs):
        net.train()
        for data, x0 in datas:
            t = torch.randint(1, T + 1, (1,))
            data["variable"].x = corrupt(x0, t, torch.randn_like(x0), ALPHA_BAR)
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()
            opt.step()
    net.eval()
    return net


def eval_k(k, net, test, K, rep):
    chk = Checker()
    c = {"langevin": 0, "random_restart": 0, "learned": 0}
    with torch.no_grad():
        for j, (g, sol) in enumerate(test):
            nv = 2 * k
            base = random.Random(100 + 101 * k + 7919 * j)
            x0 = [base.gauss(0, INIT_SD) for _ in range(nv)]
            c["langevin"] += any(
                accepted(chk, g, refine(g, x0, seed=s + K * rep, **LANGEVIN).x)
                for s in range(K))
            solved = False
            for s in range(K):
                r = random.Random(9000 * s + 31 * j + k)
                xr = [r.gauss(0, INIT_SD) for _ in range(nv)]
                if accepted(chk, g, refine(g, xr, seed=0, **POLISH).x):
                    solved = True
                    break
            c["random_restart"] += solved
            data = build_heterodata(g)
            solved = False
            for s in range(K):
                torch.manual_seed(1000 * s + 31 * j + k)
                data["variable"].x = torch.randn(nv, 1)
                prop = (net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist()
                if not all(abs(v) < 1e6 for v in prop):
                    continue
                if accepted(chk, g, refine(g, prop, seed=0, **POLISH).x):
                    solved = True
                    break
            c["learned"] += solved
    return {m: (c[m], len(test)) for m in c}


def run(ks, trials, K, epochs, ntrain, seeds):
    per_rep = []
    for rep in range(seeds):
        off = 1_000_003 * rep
        by_k = {}
        for k in ks:
            train = suite(k, ntrain, seed0=100 + k + off)
            test = suite(k, trials, seed0=90000 + k + off)
            net = train_x0(train, epochs, seed=rep)
            by_k[k] = eval_k(k, net, test, K, rep)
            print(f"  seed {rep} k={k} n={2*k}: " + "  ".join(
                f"{m}={v[0]}/{v[1]}" for m, v in by_k[k].items()), flush=True)
        per_rep.append(by_k)

    rows = []
    for k in ks:
        row = {"points": k, "n": 2 * k}
        for m in ("langevin", "random_restart", "learned"):
            ks_ = [per_rep[r][k][m][0] for r in range(seeds)]
            ts_ = [per_rep[r][k][m][1] for r in range(seeds)]
            kk, tt = sum(ks_), sum(ts_)
            cell = {"k": kk, "n": tt, "rate": kk / tt, "ci95": wilson_interval(kk, tt)}
            if seeds > 1:
                sr = [a / b for a, b in zip(ks_, ts_)]
                cell["seed_rates"] = sr
                cell["seed_mean"], cell["seed_sd"] = statistics.mean(sr), statistics.pstdev(sr)
            row[m] = cell
        _, row["p_learned_gt_random"] = two_proportion_z(
            row["learned"]["k"], row["learned"]["n"],
            row["random_restart"]["k"], row["random_restart"]["n"])
        rows.append(row)
    return {"K": K, "trials": trials, "epochs": epochs, "ntrain": ntrain, "seeds": seeds,
            "family": "make_point_chain (coupled geometry, quartic energy)",
            "methodology": "R5 dimension-scaling: per-k inline x0 training, one-shot "
                           "proposal + geometry polish, best-of-K, Checker gate",
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="trained denoiser vs random on point-chain geometry")
    ap.add_argument("--trials", type=int, default=40, help="held-out instances per k")
    ap.add_argument("--K", type=int, default=8, help="best-of-K budget (all arms)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ntrain", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ks = [1, 2] if args.quick else KS
    if args.quick:
        args.epochs, args.ntrain, args.trials = 20, 40, 10
    print(f"Point-chain LEARNED eval — best-of-{args.K}, {args.trials} test/k, "
          f"epochs={args.epochs}, ntrain={args.ntrain}, seeds={args.seeds}, ks={ks}")
    payload = run(ks, args.trials, args.K, args.epochs, args.ntrain, args.seeds)

    print(f"\n{'pts':>4} {'n':>3} {'langevin':>9} {'random':>8} {'learned':>8} {'p(l>rand)':>10}")
    for r in payload["rows"]:
        print(f"{r['points']:>4} {r['n']:>3} {r['langevin']['rate']:>9.3f} "
              f"{r['random_restart']['rate']:>8.3f} {r['learned']['rate']:>8.3f} "
              f"{r['p_learned_gt_random']:>10.4f}")
    n_win = sum(r["p_learned_gt_random"] < 0.05 and r["learned"]["rate"] > r["random_restart"]["rate"]
                for r in payload["rows"])
    print(f"\nlearned significantly beats random on {n_win}/{len(payload['rows'])} chain lengths"
          f" -> {'law prediction CONFIRMED on geometry' if n_win >= 1 else 'law prediction did NOT transfer'}")
    out = Path("results/p_geometry/pointchain_learned.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
