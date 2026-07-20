"""Dimension-scaling experiment (H1): does the LEARNED diffusion model do per-instance
inference that beats BOTH classical refinement AND a trivial prior on high-dimensional
non-convex problems — and does the advantage hold as dimension grows?

Family: n non-convex traps bundled into one problem. Each factor
r_i = (x_i - R_i)((x_i - m_i)^2 + h_i) / 15 has its only real root at R_i = +-U[3,8]
(wide magnitude AND random sign, so the solution genuinely varies per instance) and a
spurious energy basin near m_i ~ 0 where the start sits. The /15 keeps the energy gentle
enough for the shared polisher to converge. Reaching the solution requires crossing the
barrier for EVERY variable.

Methods share the ONE project-wide solver ``marc.refine.iterative.refine`` (exactly as
run_hard_eval.py / run_coupled_eval.py invoke it) and differ ONLY in the starting point:
  * deterministic : adversarial start, noise=False            -> trapped (0)
  * langevin      : adversarial start, noise=True, best-of-K  -> must escape n barriers
  * mean_prior    : CONTROL, start = training-mean solution   -> ~0 here: mean of +-roots is 0,
                    which sits AT the barrier, so a constant guess cannot work
  * random_restart: K uniform starts + polish (no learning)   -> the amortization control
  * learned_x0    : GraphDenoiser (x0 target) proposes, refine polishes, best-of-K

Acceptance is the same two-stage Checker gate the other counting scripts use. The family's
roots live on a 6-decimal grid (``make_problem`` rounds every constant), so the polished
candidate is snapped to that grid before verification — the exact analogue of the integer
solutions of the hard/coupled families snapping via the checker's rational gate. Every
rate carries a 95% Wilson CI; learned-vs-baseline comparisons carry two-proportion
z-test p-values (house rules, paper/RESULTS.md).

Outputs: results/p_scaling/scaling.json and paper/figures/fig_dimension_scaling.pdf.
Run:  python scripts/run_dimension_scaling.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from marc.graph.schema import VariableNode, FactorNode, Edge
from marc.graph.graph import FactorGraph
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.eval.metrics import wilson_interval, two_proportion_z
from marc.refine.iterative import refine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 8.0     # normalize solutions (~[-8,8]) toward unit variance
C = 15.0        # residual scaling: gentle energy so the polisher converges
DECIMALS = 6    # every family constant is generated on this decimal grid

METHODS = ["deterministic", "langevin", "mean_prior", "random_restart", "learned_x0"]

NOTE = ("unified-v2: solver and acceptance unified with run_hard_eval/run_coupled_eval "
        "(shared marc.refine.iterative.refine + Checker gate). Numbers are NOT comparable "
        "to pre-change R6 rows, which used a bespoke line-search descent and solution-space "
        "acceptance |x_i - R_i| < 0.05.")


def make_problem(n: int, rng: random.Random):
    vs, fs, es, sol, init = [], [], [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(3, 8), DECIMALS)
        m = round(rng.uniform(-0.2, 0.2), DECIMALS)
        h = round(rng.uniform(0.1, 0.3), DECIMALS)
        vs.append(VariableNode(f"x{i}"))
        fs.append(FactorNode(f"eq{i}", f"((x{i} - ({R})) * ((x{i} - ({m}))**2 + ({h}))) / {C}"))
        es.append(Edge(f"x{i}", f"eq{i}", 1))
        sol.append(R)
        init.append(round(m + rng.uniform(-0.15, 0.15), DECIMALS))
    return FactorGraph(variables=vs, factors=fs, edges=es), sol, init


def suite(n: int, count: int, seed: int):
    return [make_problem(n, random.Random(seed + 7919 * j)) for j in range(count)]


def accepted(chk: Checker, g, x) -> bool:
    """Two-stage checker gate on the candidate snapped to the family's 6-decimal grid.

    The hard/coupled families have integer solutions, which the checker's rational
    snap recovers exactly from the polished float; this family's roots are 6-decimal
    by construction, so the same idea needs the explicit grid snap first."""
    return chk.verify(g, [round(v, DECIMALS) for v in x]).accepted


def train_x0(items, epochs: int, D: int = 128, L: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    net = GraphDenoiser(D=D, L=L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), torch.tensor([[v] for v in sol], dtype=torch.float32) / SCALE)
             for g, sol, _ in items]
    for _ in range(epochs):
        net.train()
        for data, x0 in datas:
            t = torch.randint(1, T + 1, (1,))
            eps = torch.randn_like(x0)
            data["variable"].x = corrupt(x0, t, eps, ALPHA_BAR)
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()   # x0 target
            opt.step()
    net.eval()
    return net


def count_methods(test, net, tmean: float, K: int, rep: int):
    """Solved counts per method on one test suite. Returns {method: (k, n)}."""
    chk = Checker()
    counts = dict.fromkeys(METHODS, 0)
    with torch.no_grad():
        for g, sol, init in test:
            nv = len(sol)
            counts["deterministic"] += accepted(chk, g, refine(g, init, noise=False, seed=0).x)
            counts["langevin"] += any(
                accepted(chk, g, refine(g, init, noise=True, seed=s + K * rep).x)
                for s in range(K))
            counts["mean_prior"] += accepted(chk, g, refine(g, [tmean] * nv, noise=False, seed=0).x)
            # random multi-start + polish control: the baseline that isolates the learned
            # proposal's value. Learned only beats this in high dimension (amortization).
            solved = False
            for s in range(K):
                r = random.Random(9000 * s + nv + 31 * rep)
                x0 = [r.uniform(-8, 8) for _ in range(nv)]
                if accepted(chk, g, refine(g, x0, noise=False, seed=0).x):
                    solved = True
                    break
            counts["random_restart"] += solved
            data = build_heterodata(g)
            solved = False
            for s in range(K):
                torch.manual_seed(1000 * s + nv + 31 * rep)
                data["variable"].x = torch.randn(nv, 1)
                prop = (net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist()
                if accepted(chk, g, refine(g, prop, noise=False, seed=0).x):
                    solved = True
                    break
            counts["learned_x0"] += solved
    return {m: (counts[m], len(test)) for m in METHODS}


def run(ns, K: int, ntest: int, epochs: int, ntrain: int, seeds: int) -> dict:
    """Full experiment across dimensions and seed replicates; returns the JSON payload."""
    per_rep = []                     # per_rep[rep][n] = {method: (k, n)}
    for rep in range(seeds):
        off = 1_000_003 * rep        # fresh train/test draws + fresh torch init per replicate
        by_n = {}
        for n in ns:
            train = suite(n, ntrain, seed=100 + n + off)
            test = suite(n, ntest, seed=90000 + n + off)
            net = train_x0(train, epochs, seed=rep)
            tmean = sum(v for _, sol, _ in train for v in sol) / sum(len(sol) for _, sol, _ in train)
            by_n[n] = count_methods(test, net, tmean, K, rep)
            print(f"  seed {rep} n={n}: " + "  ".join(
                f"{m}={k}/{t}" for m, (k, t) in by_n[n].items()), flush=True)
        per_rep.append(by_n)

    rows = []
    for n in ns:
        row = {"n": n}
        for m in METHODS:
            ks = [per_rep[rep][n][m][0] for rep in range(seeds)]
            ts = [per_rep[rep][n][m][1] for rep in range(seeds)]
            k, t = sum(ks), sum(ts)
            cell = {"k": k, "n": t, "rate": k / t, "ci95": wilson_interval(k, t)}
            if seeds > 1:
                seed_rates = [ki / ti for ki, ti in zip(ks, ts)]
                cell["seed_rates"] = seed_rates
                cell["seed_mean"] = statistics.mean(seed_rates)
                cell["seed_std"] = statistics.stdev(seed_rates)
            row[m] = cell
        _, row["p_learned_gt_random"] = two_proportion_z(
            row["learned_x0"]["k"], row["learned_x0"]["n"],
            row["random_restart"]["k"], row["random_restart"]["n"])
        _, row["p_learned_gt_langevin"] = two_proportion_z(
            row["learned_x0"]["k"], row["learned_x0"]["n"],
            row["langevin"]["k"], row["langevin"]["n"])
        rows.append(row)

    return {"K": K, "test_per_n": ntest, "epochs": epochs, "seeds": seeds,
            "methodology": "unified-v2", "note": NOTE, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="H1 dimension-scaling: learned inference vs. classical + prior")
    ap.add_argument("--quick", action="store_true", help="tiny run for CI (n=[1,2], few epochs)")
    ap.add_argument("--K", type=int, default=8, help="best-of-K budget (all methods)")
    ap.add_argument("--test", type=int, default=40, help="held-out problems per n")
    ap.add_argument("--seeds", type=int, default=1, help="seed replicates of the whole experiment")
    args = ap.parse_args()

    ns = [1, 2] if args.quick else [1, 2, 3, 4, 6]
    epochs = 20 if args.quick else 200
    ntrain = 40 if args.quick else 200

    print(f"H1 dimension scaling — best-of-{args.K}, {args.test} test/n, epochs={epochs}, "
          f"seeds={args.seeds} (unified-v2: shared refine + Checker gate, Wilson CIs)")
    payload = run(ns, args.K, args.test, epochs, ntrain, args.seeds)

    print(f"\n{'n':>3} {'determ':>8} {'langevin':>9} {'mean_prior':>11} {'random':>8} "
          f"{'learned_x0':>11} {'p(l>rand)':>10}")
    for r in payload["rows"]:
        print(f"{r['n']:>3} {r['deterministic']['rate']:>8.3f} {r['langevin']['rate']:>9.3f} "
              f"{r['mean_prior']['rate']:>11.3f} {r['random_restart']['rate']:>8.3f} "
              f"{r['learned_x0']['rate']:>11.3f} {r['p_learned_gt_random']:>10.4f}")

    out_dir = Path("results/p_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir/'scaling.json'}")
    _plot(payload["rows"])


def _plot(rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for key, mk, col, lab in [
        ("deterministic", "o", "#888888", "deterministic"),
        ("langevin", "s", "#d62728", "Langevin (noise)"),
        ("mean_prior", "^", "#2ca02c", "mean prior"),
        ("random_restart", "v", "#9467bd", "random restart"),
        ("learned_x0", "D", "#1f77b4", "learned (ours)"),
    ]:
        rates = [r[key]["rate"] for r in rows]
        errs = [[r[key]["rate"] - r[key]["ci95"][0] for r in rows],
                [r[key]["ci95"][1] - r[key]["rate"] for r in rows]]
        ax.errorbar(ns, rates, yerr=errs, marker=mk, color=col, label=lab,
                    linewidth=2, capsize=3)
    ax.set_xlabel("problem dimension n")
    ax.set_ylabel("solve rate (best-of-K)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Learned inference vs. classical refinement & prior (95% CI)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_dimension_scaling.pdf")
    print(f"wrote {fig_dir/'fig_dimension_scaling.pdf'}")


if __name__ == "__main__":
    main()
