"""Dimension-scaling experiment (H1): does the LEARNED diffusion model do per-instance
inference that beats BOTH classical refinement AND a trivial prior on high-dimensional
non-convex problems — and does the advantage hold as dimension grows?

Family: n non-convex traps bundled into one problem. Each factor
r_i = (x_i - R_i)((x_i - m_i)^2 + h_i) / 15 has its only real root at R_i = +-U[3,8]
(wide magnitude AND random sign, so the solution genuinely varies per instance) and a
spurious energy basin near m_i ~ 0 where the start sits. The /15 keeps the energy gentle
enough for the shared polisher to converge. Reaching the solution requires crossing the
barrier for EVERY variable.

Methods share ONE local solver (backtracking line-search energy descent, optional
annealed Langevin noise) and differ ONLY in the starting point (maximally fair):
  * deterministic : adversarial start, no noise              -> trapped (0)
  * langevin      : adversarial start, annealed noise         -> must escape n barriers (~p^n)
  * mean_prior    : CONTROL, start = training-mean solution   -> ~0 here: mean of +-roots is 0,
                    which sits AT the barrier, so a constant guess cannot work
  * learned_x0    : GraphDenoiser (x0 target) predicts each root from the graph, polish

Acceptance is SOLUTION-SPACE (|x_i - R_i| < tol): scaling-invariant and the honest metric
for "did we recover the solution" (a residual tolerance is slope-dependent and unfair
across families). Disclosed in paper/dimension_scaling_result.md.

Outputs: results/p_scaling/scaling.json and paper/figures/fig_dimension_scaling.pdf.
Run:  python scripts/run_dimension_scaling.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from marc.graph.schema import VariableNode, FactorNode, Edge
from marc.graph.graph import FactorGraph
from marc.graph.serialize import save_graph
from marc.graph.pyg import build_heterodata
from marc.cas.engine import CASEngine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 8.0     # normalize solutions (~[-8,8]) toward unit variance
C = 15.0        # residual scaling: gentle energy so the polisher converges
TOL = 0.05      # solution-space acceptance |x_i - R_i| < TOL


def make_problem(n: int, rng: random.Random):
    vs, fs, es, sol, init = [], [], [], [], []
    for i in range(n):
        R = round(rng.choice([-1, 1]) * rng.uniform(3, 8), 6)
        m = round(rng.uniform(-0.2, 0.2), 6)
        h = round(rng.uniform(0.1, 0.3), 6)
        vs.append(VariableNode(f"x{i}"))
        fs.append(FactorNode(f"eq{i}", f"((x{i} - ({R})) * ((x{i} - ({m}))**2 + ({h}))) / {C}"))
        es.append(Edge(f"x{i}", f"eq{i}", 1))
        sol.append(R)
        init.append(round(m + rng.uniform(-0.15, 0.15), 6))
    return FactorGraph(variables=vs, factors=fs, edges=es), sol, init


def suite(n: int, count: int, seed: int):
    return [make_problem(n, random.Random(seed + 7919 * j)) for j in range(count)]


def cas_for(graph):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    save_graph(graph, path)
    return CASEngine(path, [v.id for v in graph.variables])


def descend(cas, x0, steps=1500, noise=False, sigma0=1.5, seed=0):
    """Shared local solver: backtracking line-search energy descent, optional annealed
    Langevin noise. No overshoot (line search) so it converges from a good start yet
    stays put in a spurious basin from a bad start."""
    rng = random.Random(seed)
    x = list(map(float, x0))
    for k in range(steps):
        if noise:
            s = sigma0 * (1 - k / steps)
            x = [xi + rng.gauss(0, s) for xi in x]
        g = cas.energy_grad(x)
        t, e0 = 1.0, cas.energy(x)
        for _ in range(60):
            xn = [xi - t * gi for xi, gi in zip(x, g)]
            if cas.energy(xn) < e0:
                x = xn
                break
            t *= 0.5
    return x


def close(x, sol, tol=TOL):
    return all(abs(a - b) < tol for a, b in zip(x, sol))


def train_x0(items, epochs: int, D: int = 128, L: int = 4):
    torch.manual_seed(0)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="H1 dimension-scaling: learned inference vs. classical + prior")
    ap.add_argument("--quick", action="store_true", help="tiny run for CI (n=[1,2], few epochs)")
    ap.add_argument("--K", type=int, default=8, help="best-of-K budget (all methods)")
    ap.add_argument("--test", type=int, default=40, help="held-out problems per n")
    args = ap.parse_args()

    ns = [1, 2] if args.quick else [1, 2, 3, 4, 6]
    epochs = 20 if args.quick else 200
    ntrain = 40 if args.quick else 200

    print(f"H1 dimension scaling — best-of-{args.K}, {args.test} test/n, epochs={epochs}")
    print(f"{'n':>3} {'determ':>8} {'langevin':>9} {'mean_prior':>11} {'random':>8} {'learned_x0':>11}")
    rows = []
    for n in ns:
        train = suite(n, ntrain, seed=100 + n)
        test = suite(n, args.test, seed=90000 + n)
        net = train_x0(train, epochs)
        tmean = sum(v for _, sol, _ in train for v in sol) / sum(len(sol) for _, sol, _ in train)
        tec = [(g, sol, init, cas_for(g)) for g, sol, init in test]

        det = sum(close(descend(c, init), sol) for g, sol, init, c in tec) / len(tec)
        lang = sum(any(close(descend(c, init, noise=True, seed=s), sol) for s in range(args.K))
                   for g, sol, init, c in tec) / len(tec)
        mean = sum(close(descend(c, [tmean] * len(sol)), sol) for g, sol, init, c in tec) / len(tec)
        # random multi-start + polish control: the baseline that isolates the learned
        # proposal's value. Learned only beats this in high dimension (the amortization result).
        import random as _rnd
        rand = 0
        for g, sol, init, c in tec:
            nv = len(sol); solved = False
            for s in range(args.K):
                rr = _rnd.Random(9000 * s + nv)
                if close(descend(c, [rr.uniform(-8, 8) for _ in range(nv)]), sol):
                    solved = True; break
            rand += int(solved)
        rand = rand / len(tec)
        okL = 0
        with torch.no_grad():
            for g, sol, init, c in tec:
                data = build_heterodata(g)
                nv = len(sol)
                solved = False
                for s in range(args.K):
                    torch.manual_seed(1000 * s + nv)
                    data["variable"].x = torch.randn(nv, 1)
                    prop = (net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist()
                    if close(descend(c, prop), sol):
                        solved = True
                        break
                okL += int(solved)
        learned = okL / len(tec)

        row = {"n": n, "deterministic": det, "langevin": lang, "mean_prior": mean,
               "random_restart": rand, "learned_x0": learned}
        rows.append(row)
        print(f"{n:>3} {det:>8.3f} {lang:>9.3f} {mean:>11.3f} {rand:>8.3f} {learned:>11.3f}", flush=True)

    out_dir = Path("results/p_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling.json").write_text(
        json.dumps({"K": args.K, "test_per_n": args.test, "epochs": epochs,
                    "tol": TOL, "rows": rows}, indent=2))
    print(f"\nwrote {out_dir/'scaling.json'}")
    _plot(rows)


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
        ax.plot(ns, [r[key] for r in rows], marker=mk, color=col, label=lab, linewidth=2)
    ax.set_xlabel("problem dimension n")
    ax.set_ylabel("solve rate (best-of-K)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Learned inference vs. classical refinement & prior")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_dimension_scaling.pdf")
    print(f"wrote {fig_dir/'fig_dimension_scaling.pdf'}")


if __name__ == "__main__":
    main()
