"""Hard-suite eval (A1) + hybrid-vs-refine ablation (A8.1).

The convex suite is saturated (every solver ~1.000), so H1 has no signal. This runs
the non-convex bilinear families (`marc.data.templates.HARD_TEMPLATES`), where
deterministic descent is trapped, and isolates the learned denoiser's contribution:

  * refine_cold      — energy descent from a cold (zero) start, best-of-K
  * refine_langevin  — annealed-noise descent from the cold start, best-of-K
  * learned_hybrid   — GraphDenoiser proposes x0, then refine() polishes it, best-of-K

If learned_hybrid > refine_* from the SAME budget, the diffusion proposal is buying
real solving power (the paper's central claim, cleanly demonstrated). This directly
answers the "what does the learned denoiser add over refine?" review attack (A8.1).

Run:  python scripts/run_hard_eval.py [--quick]
Writes results/p_hard/hard_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from marc.data.templates import HARD_TEMPLATES
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.refine.iterative import refine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 5.0  # bilinear solutions are integers in [-3,3]


def gen(template, count, seed0):
    out = []
    for i in range(count):
        g, sol = template.generate(seed=seed0 + i)
        out.append((g, [float(v) for v in sol.values()]))
    return out


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


def refine_rate(items, noise, K):
    chk = Checker()
    ok = 0
    for g, sol in items:
        solved = False
        for s in range(1 if not noise else K):
            tr = refine(g, [0.0] * len(sol), noise=noise, seed=s)
            if chk.verify(g, tr.x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok / len(items)


def hybrid_rate(items, net, K):
    """Learned proposal -> refine polish, best-of-K."""
    chk = Checker()
    ok = 0
    with torch.no_grad():
        for g, sol in items:
            data = build_heterodata(g)
            nv = len(sol)
            solved = False
            for s in range(K):
                torch.manual_seed(1000 * s + nv)
                data["variable"].x = torch.randn(nv, 1)
                prop = (net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist()
                tr = refine(g, prop, noise=False, seed=0)
                if chk.verify(g, tr.x).accepted:
                    solved = True
                    break
            ok += int(solved)
    return ok / len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 hard-suite eval + A8.1 hybrid ablation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=40)
    args = ap.parse_args()

    epochs = 20 if args.quick else 250
    ntrain = 60 if args.quick else 300

    print(f"Hard-suite eval — best-of-{args.K}, {args.test} test/family")
    print(f"{'family':18s} {'refine_cold':>12} {'refine_langevin':>16} {'learned_hybrid':>15}")
    rows = []
    for template in HARD_TEMPLATES:
        train = gen(template, ntrain, seed0=0)
        test = gen(template, args.test, seed0=100000)
        net = train_x0(train, epochs)
        row = {
            "family": template.name,
            "refine_cold": refine_rate(test, False, args.K),
            "refine_langevin": refine_rate(test, True, args.K),
            "learned_hybrid": hybrid_rate(test, net, args.K),
        }
        rows.append(row)
        print(f"{row['family']:18s} {row['refine_cold']:>12.3f} "
              f"{row['refine_langevin']:>16.3f} {row['learned_hybrid']:>15.3f}", flush=True)

    out_dir = Path("results/p_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hard_eval.json").write_text(
        json.dumps({"K": args.K, "test_per_family": args.test, "epochs": epochs, "rows": rows}, indent=2))
    print(f"\nwrote {out_dir/'hard_eval.json'}")


if __name__ == "__main__":
    main()
