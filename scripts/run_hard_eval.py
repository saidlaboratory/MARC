"""Hard-suite eval (A1) + hybrid-vs-refine ablation (A8.1), with Wilson 95% CIs.

The convex suite is saturated (every solver ~1.000), so H1 has no signal. This runs
the non-convex families (`marc.data.templates.HARD_TEMPLATES_EXT`), where deterministic
descent is trapped, and isolates the learned denoiser's contribution:

  * refine_cold      — energy descent from a cold (zero) start, best-of-K
  * refine_langevin  — annealed-noise descent from the cold start, best-of-K
  * learned_hybrid   — GraphDenoiser proposes x0, then refine() polishes it, best-of-K
  * lm               — classical scipy Levenberg–Marquardt (analytic Jacobian),
                       K Gaussian multi-starts, best-of-K (the "why not Newton/LM?" column)

If learned_hybrid's CI is disjoint from (above) refine_langevin's, the diffusion proposal
is a statistically significant win — the paper's central claim (answers "what does the
learned denoiser add over refine?", A8.1).

The learned arm self-trains a toy x0 net per family by default; pass --ckpt (or set
MARC_CKPT) to use a trained Stage-A checkpoint through the LearnedSolver DDIM path
instead — same polish and acceptance either way.

Run:  python scripts/run_hard_eval.py [--quick] [--ckpt checkpoints/denoiser_stage_a.pt]
Writes results/p_hard/hard_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from marc.data.templates import HARD_TEMPLATES_EXT
from marc.graph.pyg import build_heterodata
from marc.cas.checker import Checker
from marc.eval.metrics import rate_cell, two_proportion_z, wilson_interval
from marc.eval.runner import Problem
from marc.eval.solver import ScipySolver, load_solver
from marc.refine.iterative import refine
from marc.train.toy_x0 import gen, train_x0

T = 1000
SCALE = 5.0  # bilinear/quadratic solutions are integers in [-3,3]


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


def hybrid_count_ckpt(items, solver, K):
    # same polish + exact-accept as hybrid_count; only the proposer differs
    # (trained Stage-A checkpoint via the DDIM rollout instead of the self-trained x0 net)
    chk = Checker()
    ok = 0
    for g, sol in items:
        nv = len(sol)
        solved = False
        for s in range(K):
            torch.manual_seed(1000 * s + nv)
            prop = solver.sample(Problem(id="hyb", graph=g, solution=list(sol)), 1)[0]
            if prop is None:
                continue
            if chk.verify(g, refine(g, prop, noise=False, seed=0).x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok, len(items)


def random_count(items, K, lo=-5.0, hi=5.0):
    """Control: K random starts + deterministic polish, best-of-K (no learning).
    Isolates whether the learned proposal beats blind multi-start. On these
    small-solution families it does not — random restart ties/beats learned."""
    import random as _rnd
    chk = Checker()
    ok = 0
    for g, sol in items:
        nv = len(sol)
        solved = False
        for s in range(K):
            r = _rnd.Random(7000 * s + nv)
            x0 = [r.uniform(lo, hi) for _ in range(nv)]
            if chk.verify(g, refine(g, x0, noise=False, seed=0).x).accepted:
                solved = True
                break
        ok += int(solved)
    return ok, len(items)


def lm_count(items, K):
    """Classical baseline: scipy Levenberg–Marquardt with the analytic Jacobian,
    K independent Gaussian multi-starts, best-of-K — the "why not Newton/LM?"
    column. ExactLinearSolver is registered too, but these families are nonlinear
    so it returns no candidates; it gets no column here."""
    chk = Checker()
    solver = ScipySolver(seed=0)
    ok = 0
    for g, sol in items:
        cands = solver.sample(Problem(id="lm", graph=g, solution=list(sol)), K)
        ok += int(any(chk.verify(g, c).accepted for c in cands))
    return ok, len(items)


def _timed(count_fn, *args):
    # wall-clock the whole attempt loop (proposal + polish + accept) of one arm
    t0 = time.perf_counter()
    kn = count_fn(*args)
    return kn, (time.perf_counter() - t0) * 1000.0


def _cell(kn, ms):
    k, n = kn
    return {**rate_cell(k, n), "wall_ms_total": ms, "wall_ms_mean": ms / n}


def _fmt(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k/n:.3f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 hard-suite eval + A8.1 hybrid ablation (Wilson CIs)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--ckpt", default=os.environ.get("MARC_CKPT"),
                    help="trained checkpoint for the learned arm (defaults to $MARC_CKPT); "
                         "omit to self-train the toy x0 net as before")
    args = ap.parse_args()

    epochs = 20 if args.quick else 250
    ntrain = 60 if args.quick else 300
    families = HARD_TEMPLATES_EXT[:2] if args.quick else HARD_TEMPLATES_EXT

    if args.ckpt:
        # polish=False: hybrid_count_ckpt applies the same noise-free refine polish
        # as the self-train arm, so the two learned modes stay comparable
        solver = load_solver("learned", checkpoint=args.ckpt, polish=False)
        learned_mode = f"ckpt:{Path(args.ckpt).name}"
    else:
        solver = None
        learned_mode = "selftrain"

    print(f"Hard-suite eval — best-of-{args.K}, {args.test} test/family, 95% Wilson CIs, learned arm: {learned_mode}")
    print(f"{'family':18s} {'refine_cold':>18} {'refine_langevin':>22} {'lm':>18} {'learned_hybrid':>22} {'sig?':>5}")
    rows = []
    for template in families:
        test = gen(template, args.test, seed0=100000)
        c_cold, ms_cold = _timed(refine_count, test, False, args.K)
        c_lang, ms_lang = _timed(refine_count, test, True, args.K)
        c_rand, ms_rand = _timed(random_count, test, args.K)
        c_lm, ms_lm = _timed(lm_count, test, args.K)
        if solver is not None:
            c_hyb, ms_hyb = _timed(hybrid_count_ckpt, test, solver, args.K)
        else:
            net = train_x0(gen(template, ntrain, seed0=0), epochs)
            c_hyb, ms_hyb = _timed(hybrid_count, test, net, args.K)
        # significant if learned lower-CI > langevin upper-CI
        hyb_lo = wilson_interval(*c_hyb)[0]
        lang_hi = wilson_interval(*c_lang)[1]
        sig = hyb_lo > lang_hi
        row = {
            "family": template.name,
            "refine_cold": _cell(c_cold, ms_cold),
            "refine_langevin": _cell(c_lang, ms_lang),
            "random_restart": _cell(c_rand, ms_rand),
            "lm": _cell(c_lm, ms_lm),
            "learned_hybrid": _cell(c_hyb, ms_hyb),
            "hybrid_beats_langevin_sig": bool(sig),
            "p_learned_gt_lm": two_proportion_z(c_hyb[0], c_hyb[1], c_lm[0], c_lm[1])[1],
        }
        rows.append(row)
        print(f"{template.name:18s} cold={_fmt(*c_cold)}  lang={_fmt(*c_lang)}  "
              f"random={_fmt(*c_rand)}  lm={_fmt(*c_lm)}  learned={_fmt(*c_hyb)}", flush=True)

    n_sig = sum(r["hybrid_beats_langevin_sig"] for r in rows)
    print(f"\nlearned_hybrid CI-disjoint above refine_langevin on {n_sig}/{len(rows)} families")
    out_dir = Path("results/p_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hard_eval.json").write_text(
        json.dumps({"K": args.K, "test_per_family": args.test, "epochs": epochs,
                    "learned_mode": learned_mode,
                    "n_significant": n_sig, "rows": rows}, indent=2))
    print(f"wrote {out_dir/'hard_eval.json'}")


if __name__ == "__main__":
    main()
