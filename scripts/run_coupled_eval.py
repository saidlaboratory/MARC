"""Coupled high-dimensional constraint solving (the main-track gate).

Independent bundled traps let a learned model "memorize each variable's marginal root",
and random restart collapses in high dim (must hit all n basins at once) — so the earlier
crossover was partly an independence artifact. This uses a COUPLED chained bilinear system:

    x_i + x_{i+1} = s_i,   x_i * x_{i+1} = p_i    (i = 0 .. n-2)

The solution is a joint object (not a product of per-variable marginals), and — key —
random restart + polish does NOT collapse here (~0.4 at all n), because the chain lets the
polish propagate. So the learned model must be *genuinely* better than a strong random
baseline, not just win because random fails. If it clears that bar at high n, it is doing
real joint amortized inference (a main-track-relevant result). If it ties random, the
earlier advantage was an independence artifact — reported honestly either way.

Also reports an ``lm`` column: classical scipy Levenberg–Marquardt with the analytic
Jacobian, K Gaussian multi-starts, best-of-K — the classical-solver baseline.

Run:  python scripts/run_coupled_eval.py [--quick]
Writes results/p_coupled/coupled.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from marc.data.coupled import make_chain
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.eval.metrics import wilson_interval, two_proportion_z
from marc.eval.runner import Problem
from marc.eval.solver import ScipySolver
from marc.refine.iterative import refine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 4.0


def gen(n, count, seed0):
    return [make_chain(n, random.Random(seed0 + i)) for i in range(count)]


def train_x0(items, epochs, D=128, L=4):
    torch.manual_seed(0)
    net = GraphDenoiser(D=D, L=L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), torch.tensor([[v] for v in sol], dtype=torch.float32) / SCALE)
             for g, sol in items]
    for _ in range(epochs):
        net.train()
        for data, x0 in datas:
            t = torch.randint(1, T + 1, (1,))
            eps = torch.randn_like(x0)
            data["variable"].x = corrupt(x0, t, eps, ALPHA_BAR)
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()
            opt.step()
    net.eval()
    return net


def langevin_count(items, K):
    chk = Checker(); ok = 0
    for g, sol in items:
        ok += any(chk.verify(g, refine(g, [0.0] * len(sol), noise=True, seed=s).x).accepted
                  for s in range(K))
    return ok, len(items)


def random_count(items, K):
    chk = Checker(); ok = 0
    for g, sol in items:
        nv = len(sol); solved = False
        for s in range(K):
            r = random.Random(9000 * s + nv)
            if chk.verify(g, refine(g, [r.uniform(-4, 4) for _ in range(nv)], noise=False).x).accepted:
                solved = True; break
        ok += int(solved)
    return ok, len(items)


def lm_count(items, K):
    """Classical baseline: scipy Levenberg–Marquardt (analytic Jacobian), K Gaussian
    multi-starts, best-of-K. The chain is nonlinear, so the registered
    ExactLinearSolver returns no candidates and gets no column here."""
    chk = Checker(); ok = 0
    solver = ScipySolver(seed=0)
    for g, sol in items:
        cands = solver.sample(Problem(id="lm", graph=g, solution=list(sol)), K)
        ok += int(any(chk.verify(g, c).accepted for c in cands))
    return ok, len(items)


def hybrid_count(items, net, K):
    chk = Checker(); ok = 0
    with torch.no_grad():
        for g, sol in items:
            data = build_heterodata(g); nv = len(sol); solved = False
            for s in range(K):
                torch.manual_seed(1000 * s + nv)
                data["variable"].x = torch.randn(nv, 1)
                prop = (net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist()
                if chk.verify(g, refine(g, prop, noise=False).x).accepted:
                    solved = True; break
            ok += int(solved)
    return ok, len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coupled chained-bilinear scaling (main-track gate)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=60)
    args = ap.parse_args()
    ns = [2, 3] if args.quick else [2, 3, 4, 6, 8]
    epochs = 20 if args.quick else 250
    ntrain = 60 if args.quick else 300

    print(f"Coupled chained bilinear — best-of-{args.K}, {args.test} test/n")
    print(f"{'n':>3} {'langevin':>9} {'random':>9} {'lm':>9} {'learned':>9} {'p(l>rand)':>10} {'p(l>lm)':>9}")
    rows = []
    for n in ns:
        train = gen(n, ntrain, seed0=0)
        test = gen(n, args.test, seed0=500000)
        net = train_x0(train, epochs)
        cl = langevin_count(test, args.K)
        cr = random_count(test, args.K)
        clm = lm_count(test, args.K)
        ch = hybrid_count(test, net, args.K)
        _, p = two_proportion_z(ch[0], ch[1], cr[0], cr[1])
        _, p_lm = two_proportion_z(ch[0], ch[1], clm[0], clm[1])
        rows.append({"n": n,
                     "langevin": {"rate": cl[0] / cl[1], "ci95": wilson_interval(*cl)},
                     "random": {"rate": cr[0] / cr[1], "ci95": wilson_interval(*cr)},
                     "lm": {"k": clm[0], "n": clm[1], "rate": clm[0] / clm[1], "ci95": wilson_interval(*clm)},
                     "learned": {"k": ch[0], "n": ch[1], "rate": ch[0] / ch[1], "ci95": wilson_interval(*ch)},
                     "p_learned_gt_random": p,
                     "p_learned_gt_lm": p_lm})
        print(f"{n:>3} {cl[0]/cl[1]:>9.3f} {cr[0]/cr[1]:>9.3f} {clm[0]/clm[1]:>9.3f} "
              f"{ch[0]/ch[1]:>9.3f} {p:>10.4f} {p_lm:>9.4f}", flush=True)

    out = Path("results/p_coupled"); out.mkdir(parents=True, exist_ok=True)
    (out / "coupled.json").write_text(json.dumps({"K": args.K, "test_per_n": args.test,
                                                  "epochs": epochs, "rows": rows}, indent=2))
    n_win = sum(r["p_learned_gt_random"] < 0.05 and r["learned"]["rate"] > r["random"]["rate"] for r in rows)
    print(f"\nlearned significantly beats random on {n_win}/{len(rows)} dims → "
          f"{'REAL joint amortized inference (main-track signal)' if n_win >= 2 else 'ties random (independence artifact)'}")


if __name__ == "__main__":
    main()
