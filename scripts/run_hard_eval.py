"""Hard-suite eval (A1) + hybrid-vs-refine ablation (A8.1), with Wilson 95% CIs.

The convex suite is saturated (every solver ~1.000), so H1 has no signal. This runs
the non-convex families (`marc.data.templates.HARD_TEMPLATES_EXT`), where deterministic
descent is trapped, and isolates the learned denoiser's contribution:

  * refine_cold      — energy descent from a cold (zero) start, best-of-K
  * refine_langevin  — annealed-noise descent from the cold start, best-of-K
  * learned_hybrid   — GraphDenoiser proposes x0, then refine() polishes it, best-of-K

If learned_hybrid's CI is disjoint from (above) refine_langevin's, the diffusion proposal
is a statistically significant win — the paper's central claim (answers "what does the
learned denoiser add over refine?", A8.1).

Run:  python scripts/run_hard_eval.py [--quick]
Writes results/p_hard/hard_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from marc.data.templates import HARD_TEMPLATES_EXT
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.eval.metrics import wilson_interval
from marc.refine.iterative import refine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 5.0  # bilinear/quadratic solutions are integers in [-3,3]


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


def refine_count(items, noise, K):
    chk = Checker()
    ok = 0
    for g, sol in items:
        solved = False
        for s in range(1 if not noise else K):
            if chk.verify(g, refine(g, [0.0] * len(sol), noise=noise, seed=s).x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok, len(items)


def hybrid_count(items, net, K):
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
                if chk.verify(g, refine(g, prop, noise=False, seed=0).x).accepted:
                    solved = True
                    break
            ok += int(solved)
    return ok, len(items)


def _fmt(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k/n:.3f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 hard-suite eval + A8.1 hybrid ablation (Wilson CIs)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=60)
    args = ap.parse_args()

    epochs = 20 if args.quick else 250
    ntrain = 60 if args.quick else 300
    families = HARD_TEMPLATES_EXT[:2] if args.quick else HARD_TEMPLATES_EXT

    print(f"Hard-suite eval — best-of-{args.K}, {args.test} test/family, 95% Wilson CIs")
    print(f"{'family':18s} {'refine_cold':>18} {'refine_langevin':>22} {'learned_hybrid':>22} {'sig?':>5}")
    rows = []
    for template in families:
        train = gen(template, ntrain, seed0=0)
        test = gen(template, args.test, seed0=100000)
        net = train_x0(train, epochs)
        c_cold = refine_count(test, False, args.K)
        c_lang = refine_count(test, True, args.K)
        c_hyb = hybrid_count(test, net, args.K)
        # significant if learned lower-CI > langevin upper-CI
        hyb_lo = wilson_interval(*c_hyb)[0]
        lang_hi = wilson_interval(*c_lang)[1]
        sig = hyb_lo > lang_hi
        row = {
            "family": template.name,
            "refine_cold": {"k": c_cold[0], "n": c_cold[1], "rate": c_cold[0] / c_cold[1],
                            "ci95": wilson_interval(*c_cold)},
            "refine_langevin": {"k": c_lang[0], "n": c_lang[1], "rate": c_lang[0] / c_lang[1],
                                "ci95": wilson_interval(*c_lang)},
            "learned_hybrid": {"k": c_hyb[0], "n": c_hyb[1], "rate": c_hyb[0] / c_hyb[1],
                               "ci95": wilson_interval(*c_hyb)},
            "hybrid_beats_langevin_sig": bool(sig),
        }
        rows.append(row)
        print(f"{template.name:18s} {_fmt(*c_cold):>18} {_fmt(*c_lang):>22} "
              f"{_fmt(*c_hyb):>22} {'YES' if sig else 'no':>5}", flush=True)

    n_sig = sum(r["hybrid_beats_langevin_sig"] for r in rows)
    print(f"\nlearned_hybrid CI-disjoint above refine_langevin on {n_sig}/{len(rows)} families")
    out_dir = Path("results/p_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hard_eval.json").write_text(
        json.dumps({"K": args.K, "test_per_family": args.test, "epochs": epochs,
                    "n_significant": n_sig, "rows": rows}, indent=2))
    print(f"wrote {out_dir/'hard_eval.json'}")


if __name__ == "__main__":
    main()
