"""Failure-mode diagnostic: why CircleLine defeats the learned proposal (R3/R4).

CircleLine's two roots are swapped by x<->y; an MSE/x0 proposal regresses to
their mean — the chord midpoint (s/2, s/2), on the line but never on the circle —
and the x=y diagonal it lands on is invariant under deterministic polish. Three
measurements: (a) root gap + midpoint energy + polish-from-midpoint, (b) where a
CircleLine-only trained proposal lands, (c) best-of-K polish from proposals vs
random inits (ties the mechanism to the observed 0.000 vs 0.200).

Run:  PYTHONPATH=. python3 scripts/diagnose_circleline.py [--n 200] [--epochs 100]
Writes results/p_hard/circleline_diag.json and paper/notes/circleline_diagnostic.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn

from marc.cas.checker import Checker
from marc.data.templates import CircleLineTemplate
from marc.diffusion.forward import corrupt
from marc.diffusion.schedule import cosine_beta_schedule
from marc.eval.metrics import wilson_interval
from marc.graph.pyg import build_heterodata
from marc.model.denoiser import GraphDenoiser
from marc.refine.iterative import build_energy_fns, refine

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)
SCALE = 5.0
TEMPLATE = CircleLineTemplate()
K = 8
NTRAIN = 300


def gen(template, count, seed0):
    out = []
    for i in range(count):
        g, sol = template.generate(seed=seed0 + i)
        out.append((g, [float(v) for v in sol.values()]))
    return out


# copied from scripts/run_hard_eval.py so the diagnostic trains the exact model it indicts
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


def true_roots(sol):
    """Both real roots of x+y=s, x^2+y^2=r. r and s are symmetric in (x, y), so
    the second root is the swap of the generated one; the closed form
    x = (s ± sqrt(2r - s^2))/2 gives exactly these two points."""
    x, y = sol
    return [(x, y), (y, x)]


def chord_midpoint(sol):
    m = (sol[0] + sol[1]) / 2.0
    return (m, m)


def energy_at(g, pt):
    e_fn, _, _ = build_energy_fns(g)
    return float(e_fn(*pt))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polished_ok(chk, g, x0):
    return chk.verify(g, refine(g, list(x0), noise=False, seed=0).x).accepted


def geometry(items):
    chk = Checker()
    gaps, mid_es = [], []
    mid_ok = 0
    for g, sol in items:
        r0, r1 = true_roots(sol)
        gaps.append(dist(r0, r1))
        mid = chord_midpoint(sol)
        mid_es.append(energy_at(g, mid))
        mid_ok += int(polished_ok(chk, g, mid))
    n = len(items)
    return {"mean_root_gap": mean(gaps), "min_root_gap": min(gaps),
            "mean_midpoint_energy": mean(mid_es), "min_midpoint_energy": min(mid_es),
            "midpoint_polish": {"k": mid_ok, "n": n, "rate": mid_ok / n,
                                "ci95": wilson_interval(mid_ok, n)}}


def propose(net, items):
    """K proposals per instance, seeded exactly as run_hard_eval.hybrid_count."""
    out = []
    with torch.no_grad():
        for g, sol in items:
            data = build_heterodata(g)
            nv = len(sol)
            per = []
            for s in range(K):
                torch.manual_seed(1000 * s + nv)
                data["variable"].x = torch.randn(nv, 1)
                per.append((net(data, torch.tensor([T])) * SCALE).reshape(-1).tolist())
            out.append(per)
    return out


def proposal_stats(items, props):
    d_root, d_mid, closer, off_diag = [], [], [], []
    for (g, sol), per in zip(items, props):
        r0, r1 = true_roots(sol)
        mid = chord_midpoint(sol)
        for p in per:
            dr = min(dist(p, r0), dist(p, r1))
            dm = dist(p, mid)
            d_root.append(dr)
            d_mid.append(dm)
            closer.append(dm < dr)
            off_diag.append(abs(p[0] - p[1]))
    return {"mean_dist_nearest_root": mean(d_root), "mean_dist_chord_midpoint": mean(d_mid),
            "frac_closer_to_midpoint": mean(closer), "mean_offdiag": mean(off_diag),
            "max_offdiag": max(off_diag)}


def learned_rate(items, props):
    chk = Checker()
    ok = sum(int(any(polished_ok(chk, g, p) for p in per))
             for (g, _), per in zip(items, props))
    return ok, len(items)


def random_rate(items, lo=-5.0, hi=5.0):
    """Same seeding as run_hard_eval.random_count: K random starts, same polish."""
    chk = Checker()
    ok = 0
    for g, sol in items:
        nv = len(sol)
        solved = False
        for s in range(K):
            r = random.Random(7000 * s + nv)
            if polished_ok(chk, g, [r.uniform(lo, hi) for _ in range(nv)]):
                solved = True
                break
        ok += int(solved)
    return ok, len(items)


def rate_cell(k, n):
    return {"k": k, "n": n, "rate": k / n, "ci95": wilson_interval(k, n)}


def write_note(path, p):
    geo, pr = p["geometry"], p["proposals"]
    lr, rr, mp = p["solve"]["learned_hybrid"], p["solve"]["random_restart"], geo["midpoint_polish"]
    path.write_text(f"""# Why the learned proposal scores 0.000 on CircleLine

Diagnostic for the R3/R4 failure (learned hybrid 0.000 vs random-init+polish 0.200
on CircleLine; adding CircleLine to the training mix also collapses BilinearSystem
transfer, R4). Produced by `PYTHONPATH=. python3 scripts/diagnose_circleline.py`
({p["n_instances"]} held-out instances, best-of-{p["K"]}, x0 model trained on
CircleLine only, {p["epochs"]} epochs x {p["ntrain"]} instances — the run_hard_eval
recipe). Full numbers in `results/p_hard/circleline_diag.json`.

## The target is bimodal and its mean is infeasible

Every CircleLine instance (x²+y²=r, x+y=s) has exactly two real roots, and they are
mirror images under x↔y: (x\\*, y\\*) and (y\\*, x\\*). Measured over
{p["n_instances"]} instances, the two roots sit {geo["mean_root_gap"]:.2f} apart on
average (never closer than {geo["min_root_gap"]:.2f}). Their mean — the chord
midpoint (s/2, s/2) — satisfies the line exactly but misses the circle on every
instance: energy E = (x\\*−y\\*)⁴/8 at the midpoint averages
{geo["mean_midpoint_energy"]:.2f} and is never below {geo["min_midpoint_energy"]:.3f}.
So the minimizer of the MSE/x0 objective, the conditional mean of the roots, is an
infeasible point by construction.

## The trained proposal lands on the diagonal, near the midpoint

The x and y nodes of a CircleLine factor graph have identical neighborhoods, so the
permutation-equivariant denoiser can only tell them apart through its noise input —
which x0-regression trains it to ignore. The measurement agrees: proposals from the
CircleLine-only model sit {pr["mean_offdiag"]:.1e} off the diagonal x=y on average
(max {pr["max_offdiag"]:.1e} — float precision; the x and y outputs are identical),
against a mean root gap of {geo["mean_root_gap"]:.2f}.
They land {pr["mean_dist_chord_midpoint"]:.2f} from the chord midpoint versus
{pr["mean_dist_nearest_root"]:.2f} from the nearest root, and
{100 * pr["frac_closer_to_midpoint"]:.0f}% of proposals are closer to the midpoint
than to either root — the regression-to-the-mean signature.

## The diagonal is in neither root's basin

The energy is symmetric under x↔y, so deterministic gradient descent started on the
diagonal stays on it, and no root lies there (roots have x≠y). Polish from the exact
midpoint reaches a root on {mp["k"]}/{mp["n"]} instances. Best-of-{p["K"]} polish
from the learned proposals solves {lr["k"]}/{lr["n"]} = {lr["rate"]:.3f}
[{lr["ci95"][0]:.2f},{lr["ci95"][1]:.2f}]; the same polish budget from random inits
in [−5,5] solves {rr["k"]}/{rr["n"]} = {rr["rate"]:.3f}
[{rr["ci95"][0]:.2f},{rr["ci95"][1]:.2f}]. Random starts break the symmetry the
proposal cannot; that asymmetry is the whole 0.000-vs-0.200 gap.

## Implication for the paper

This is the cleanest concrete instance of the central claim: an MSE/x0 proposal
learns the mean of a multimodal solution set, not its modes. CircleLine is the
worst case because its two modes are exact mirror images, the mean lies on a
symmetry axis that is invariant under the polish dynamics, and the architecture
itself cannot break the tie. It also offers a plausible reading of the R4 transfer
collapse — CircleLine gradients pull shared weights toward symmetric (diagonal)
predictions that are wrong for BilinearSystem too — though that link is untested
here. Fixes would need a multimodal proposal head (sampling the reverse chain
rather than one-shot x0, or symmetry-breaking features), not more training.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description="CircleLine failure-mode diagnostic")
    ap.add_argument("--n", type=int, default=200, help="held-out instances")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    test = gen(TEMPLATE, args.n, seed0=100000)
    geo = geometry(test)
    print(f"geometry: root_gap mean {geo['mean_root_gap']:.2f} min {geo['min_root_gap']:.2f}; "
          f"midpoint energy mean {geo['mean_midpoint_energy']:.2f} min {geo['min_midpoint_energy']:.3f}; "
          f"midpoint-polish {geo['midpoint_polish']['k']}/{geo['midpoint_polish']['n']}", flush=True)

    net = train_x0(gen(TEMPLATE, NTRAIN, seed0=0), args.epochs)
    props = propose(net, test)
    pstats = proposal_stats(test, props)
    print(f"proposals: d(nearest root) {pstats['mean_dist_nearest_root']:.2f}, "
          f"d(midpoint) {pstats['mean_dist_chord_midpoint']:.2f}, "
          f"closer-to-midpoint {pstats['frac_closer_to_midpoint']:.2f}, "
          f"|px-py| {pstats['mean_offdiag']:.3f}", flush=True)

    lk, ln = learned_rate(test, props)
    rk, rn = random_rate(test)
    print(f"solve: learned {lk}/{ln}, random {rk}/{rn}")

    payload = {"n_instances": args.n, "K": K, "epochs": args.epochs, "ntrain": NTRAIN,
               "geometry": geo, "proposals": pstats,
               "solve": {"learned_hybrid": rate_cell(lk, ln), "random_restart": rate_cell(rk, rn)}}
    out_dir = Path("results/p_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circleline_diag.json").write_text(json.dumps(payload, indent=2))
    note_dir = Path("paper/notes")
    note_dir.mkdir(parents=True, exist_ok=True)
    write_note(note_dir / "circleline_diagnostic.md", payload)
    print(f"wrote {out_dir/'circleline_diag.json'} and {note_dir/'circleline_diagnostic.md'}")


if __name__ == "__main__":
    main()
