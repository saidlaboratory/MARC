"""Cross-family generalization (H1 transfer): does a learned model trained on some
non-convex families solve a HELD-OUT family it never saw during training?

Leave-one-family-out: for each held-out family H, train the GraphDenoiser on the other
three families (mixed), then run the diffusion-proposal + refine-polish hybrid on H's
held-out instances and compare to refine+Langevin on H. If the learned hybrid still
beats the classical baseline on a family it never trained on, that is structural
generalization (H1), not memorization.

Run:  python scripts/run_crossfamily_eval.py [--quick]
Writes results/p_hard/crossfamily.json.

--ckpt (or MARC_CKPT) swaps the LOFO self-training for a pre-trained Stage-A
checkpoint via LearnedSolver's DDIM path. That checkpoint saw ALL families, so
held-out results are contaminated and NOT comparable to the LOFO protocol — the
JSON carries an explicit caveat. Use it only to ask whether the big model helps
at all on these families; the default self-train run stays canonical.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from marc.data.templates import HARD_TEMPLATES_EXT
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.eval.metrics import wilson_interval, two_proportion_z
from marc.eval.solver import LearnedSolver
from marc.refine.iterative import refine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 5.0


def gen(template, count, seed0):
    return [(g, [float(v) for v in sol.values()])
            for i in range(count)
            for g, sol in [template.generate(seed=seed0 + i)]]


def train_x0(items, epochs, D=128, L=4):
    torch.manual_seed(0)
    net = GraphDenoiser(D=D, L=L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), torch.tensor([[v] for v in sol], dtype=torch.float32) / SCALE)
             for g, sol in items]
    order = list(range(len(datas)))
    for _ in range(epochs):
        net.train()
        torch.randperm(len(order))
        for idx in order:
            data, x0 = datas[idx]
            t = torch.randint(1, T + 1, (1,))
            eps = torch.randn_like(x0)
            data["variable"].x = corrupt(x0, t, eps, ALPHA_BAR)
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()
            opt.step()
    net.eval()
    return net


def refine_count(items, noise, K):
    chk = Checker(); ok = 0
    for g, sol in items:
        solved = any(chk.verify(g, refine(g, [0.0] * len(sol), noise=noise, seed=s).x).accepted
                     for s in range(1 if not noise else K))
        ok += int(solved)
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
                if chk.verify(g, refine(g, prop, noise=False, seed=0).x).accepted:
                    solved = True; break
            ok += int(solved)
    return ok, len(items)


def learned_count(items, solver, K):
    chk = Checker(); ok = 0
    for g, sol in items:
        cands = solver.sample(SimpleNamespace(graph=g), K)
        ok += int(any(c is not None and chk.verify(g, c).accepted for c in cands))
    return ok, len(items)


def run(fams, K, ntest, epochs, ntrain, ckpt=None):
    solver = LearnedSolver(ckpt) if ckpt else None
    mode = f"ckpt:{Path(ckpt).name}" if ckpt else "selftrain"
    print(f"Leave-one-family-out generalization — best-of-{K}, {ntest} test/family [{mode}]")
    print(f"{'held-out family':18s} {'refine_langevin':>18} {'learned(cross)':>18} {'p':>8}")
    rows = []
    for i, held in enumerate(fams):
        train_fams = [f for j, f in enumerate(fams) if j != i]
        test = gen(held, ntest, seed0=100000)
        c_lang = refine_count(test, True, K)
        if solver:
            c_hyb = learned_count(test, solver, K)
        else:
            train = []
            for tf in train_fams:
                train += gen(tf, ntrain, seed0=0)
            c_hyb = hybrid_count(test, train_x0(train, epochs), K)
        z, p = two_proportion_z(c_hyb[0], c_hyb[1], c_lang[0], c_lang[1])
        rows.append({
            "held_out": held.name,
            "trained_on": "all_families_ckpt" if solver else [f.name for f in train_fams],
            "refine_langevin": {"k": c_lang[0], "n": c_lang[1], "rate": c_lang[0] / c_lang[1],
                                "ci95": wilson_interval(*c_lang)},
            "learned_cross": {"k": c_hyb[0], "n": c_hyb[1], "rate": c_hyb[0] / c_hyb[1],
                              "ci95": wilson_interval(*c_hyb)},
            "p_hybrid_gt_langevin": p,
        })
        print(f"{held.name:18s} {c_lang[0]/c_lang[1]:>18.3f} {c_hyb[0]/c_hyb[1]:>18.3f} {p:>8.4f}",
              flush=True)

    payload = {"K": K, "test_per_family": ntest, "epochs": epochs,
               "learned_mode": mode, "rows": rows}
    if ckpt:
        payload["ckpt_mode_caveat"] = (
            "checkpoint trained on all families; held-out contamination — "
            "not comparable to LOFO self-train protocol")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-family (leave-one-out) generalization")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--ckpt", default=os.environ.get("MARC_CKPT"),
                    help="Stage-A checkpoint (DDIM via LearnedSolver). Trained on all "
                         "families, so held-out results are contaminated vs LOFO.")
    args = ap.parse_args()
    epochs = 20 if args.quick else 200
    ntrain = 40 if args.quick else 200
    fams = HARD_TEMPLATES_EXT[:3] if args.quick else HARD_TEMPLATES_EXT

    payload = run(fams, args.K, args.test, epochs, ntrain, ckpt=args.ckpt)
    out_dir = Path("results/p_hard"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crossfamily.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir/'crossfamily.json'}")


if __name__ == "__main__":
    main()
