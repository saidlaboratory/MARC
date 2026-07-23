#!/usr/bin/env python3
"""Pilot: does derived-construction repair transfer to the named real systems?

Half-day external-validity probe for the construction-repair result: build
failure populations from HARDENED, parameterized variants of three of the
paper's named real system classes (marc.data.real_systems), then ask whether
the geo_repair vocabulary pattern — derived equality pins (Heron/Cayley-Menger
cross-product magnitudes, both signs) and redundant law-of-cosines lifts, all
functions of the GIVENS only — flips those failures at a rate restarts cannot.

Classes (parameterized, harder than the fixed suite entries):
  ik3r_random   3R IK: random links, random target + full (sin,cos) end
                orientation, wrist across the 2R annulus. Discrete structure:
                elbow / base-triangle branch. Honest negative: LM k=4 never
                fails here (scans over the (c,s) and joint-angle encodings,
                init scales 1-8, rim/inner-rim targets, link disparity all
                gave 0/30 two-stream failures), so there is no failure
                population to repair; kept as the documented negative.
  trilat_far    3-station trilateration, near-collinear station line OFFSET
                from the origin, true point on the far side: Gaussian inits
                land on the near side and LM sticks in the mirror local
                minimum (the classic GPS far-init failure).
  circles_far   two intersecting circles far from the origin at the stock
                init scale of the paper's circle system. Also a measured
                negative: LM travels to distant circles without local minima.
  conic_ghost   coaxial eccentric ellipse + circle, far from the origin, where
                the symmetry-axis elimination quadratic has one REAL
                intersection root and one ghost root (w^2 < 0): the ghost's
                near-tangency point is a spurious attractor for LM.

Protocol (mirrors marc.structure.geo_repair):
  * reference = REFERENCE_SOLVER (scipy LM, k=4 Gaussian multistarts); NUMERIC
    acceptance max|r| < 1e-6 over the ORIGINAL factors only (roots are
    irrational — run_real_systems.py convention). Aux factors join the
    least-squares but never the acceptance test.
  * an instance is a failure only if the direct solve fails on TWO independent
    restart streams (single-stream selection keeps noise failures).
  * every arm is graded on one fresh common stream (CRN): constructions add
    factors, never variables, so all arms see byte-identical inits and outcome
    differences come from the added factor alone.
  * controls: +K_REF fresh restarts (matched to one construction attempt) and
    +K_REF*V fresh restarts (matched to the full enumeration budget over the
    V-construction vocabulary).

Pilot only: prints the table, writes /tmp JSON, touches nothing in results/.
Run:  PYTHONPATH=. python3 scripts/pilot_real_repair.py [--n 100]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.data.real_systems import _g
from marc.eval.metrics import rate_cell
from marc.eval.solver import load_solver
from marc.structure.geo_repair import Construction, _heron_sq
from marc.structure.invention_data import REFERENCE_SOLVER

K_REF = REFERENCE_SOLVER["k_refine"]
TOL = 1e-6            # numeric acceptance, as in run_real_systems.py
SALT = 10_000_019     # purpose-stream separation, as in geo_repair

IK3R_SCALE = 1.0      # the suite's stock scale for the (c,s) parameterization
TRILAT_SCALE = 5.0    # inits near origin; the station line sits 8-14 away
CIRC_SCALE = 3.0      # stock scale of the suite's circle system; centers 7-13 away
CONIC_SCALE = 3.0     # stock conic scale; the coaxial pair sits 12-20 away


def residual_fn(graph):
    """Compiled max-|residual| over the ORIGINAL factors (run_real_systems.py)."""
    syms = [sp.Symbol(v.id) for v in graph.variables]
    fns = [sp.lambdify(syms, sp.sympify(f.expression), "math") for f in graph.factors]

    def maxres(x):
        try:
            return max(abs(fn(*x)) for fn in fns)
        except (OverflowError, ValueError):
            return float("inf")
    return maxres


def lm_solve(graph, maxres, seed, scale, k=K_REF):
    """One reference-protocol attempt; acceptance always on the original maxres."""
    solver = load_solver("lm", seed=seed, init_scale=scale)
    prob = SimpleNamespace(id="pilot", graph=graph,
                           solution=[0.0] * len(graph.variables), metadata={})
    return any(c is not None and maxres(c) < TOL for c in solver.sample(prob, k))


def _pins(name_fmt, kind, pos, expr_fmt, varnames, heron):
    """Both signs of a cross-product pin, skipped when the triangle degenerates."""
    if heron <= 1e-9:
        return []
    v = math.sqrt(heron) / 2.0
    return [Construction(name_fmt.format("p" if sgn > 0 else "m"), kind, pos, sgn,
                         expr_fmt.format(sgn * v), varnames)
            for sgn in (1.0, -1.0)]


def gen_ik3r(rng):
    L1, L2, L3 = (rng.uniform(0.6, 1.4) for _ in range(3))
    lo, hi = abs(L1 - L2), L1 + L2
    r = lo + (hi - lo) * rng.uniform(0.15, 0.97)
    psi, phi = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
    wx, wy = r * math.cos(psi), r * math.sin(psi)       # wrist = target - L3*e(phi)
    tx, ty = wx + L3 * math.cos(phi), wy + L3 * math.sin(phi)
    c12, s12 = "(c1*c2 - s1*s2)", "(s1*c2 + c1*s2)"
    c123, s123 = f"({c12}*c3 - {s12}*s3)", f"({s12}*c3 + {c12}*s3)"
    graph = _g(["c1", "s1", "c2", "s2", "c3", "s3"],
               [("u1", "c1**2 + s1**2 - 1"),
                ("u2", "c2**2 + s2**2 - 1"),
                ("u3", "c3**2 + s3**2 - 1"),
                ("px", f"{L1}*c1 + {L2}*{c12} + {L3}*{c123} - ({tx})"),
                ("py", f"{L1}*s1 + {L2}*{s12} + {L3}*{s123} - ({ty})"),
                ("oc", f"{c123} - ({math.cos(phi)})"),
                ("os", f"{s123} - ({math.sin(phi)})")], IK3R_SCALE)
    d2 = wx * wx + wy * wy
    # law-of-cosines lifts on the base/elbow triangle (L1, L2, |w|): derived,
    # branch-free; the elbow one determines c2 outright.
    vocab = [
        Construction("elbow_cos", "cos", 2, 0.0,
                     f"c2 - ({(d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)})", ("c2",)),
        Construction("base_cos", "cos", 1, 0.0,
                     f"({wx})*c1 + ({wy})*s1 - ({(d2 + L1 * L1 - L2 * L2) / (2 * L1)})",
                     ("c1", "s1")),
    ]
    # elbow branch: L1*L2*s2 = +-2*Area(L1, L2, |w|)  =>  s2 = +-sqrt(heron)/(2*L1*L2)
    s = _heron_sq(L1 * L1, L2 * L2, d2)
    if s > 1e-9:
        v = math.sqrt(s) / (2 * L1 * L2)
        vocab += [Construction(f"elbow_{'p' if sgn > 0 else 'm'}", "branch", 2, sgn,
                               f"s2 - ({sgn * v})", ("s2",)) for sgn in (1.0, -1.0)]
    # base branch: cross(w, e(theta1)) = +-2*Area(|w|, L1, L2)/L1
    vocab += _pins("base_{}", "branch", 1,
                   f"({wx})*s1 - ({wy})*c1 - ({{}})",
                   ("c1", "s1"), _heron_sq(d2, L1 * L1, L2 * L2) / (L1 * L1))
    return graph, IK3R_SCALE, vocab


def gen_trilat(rng):
    ang = rng.uniform(-math.pi, math.pi)
    nhx, nhy = math.cos(ang), math.sin(ang)          # normal: origin -> station line
    ux, uy = -nhy, nhx
    R = rng.uniform(8.0, 14.0)
    ts = (-4.0, rng.uniform(-1.5, 1.5), 4.0)
    st = [(R * nhx + t * ux + (o := rng.uniform(-0.2, 0.2)) * nhx,
           R * nhy + t * uy + o * nhy) for t in ts]
    a = rng.uniform(-3.5, 3.5)
    h = rng.uniform(1.5, 3.5)                        # true point BEYOND the line
    px, py = R * nhx + a * ux + h * nhx, R * nhy + a * uy + h * nhy
    d2 = [(px - sx) ** 2 + (py - sy) ** 2 for sx, sy in st]
    graph = _g(["x", "y"],
               [(f"a{i}", f"(x - ({sx}))**2 + (y - ({sy}))**2 - ({d2[i]})")
                for i, (sx, sy) in enumerate(st)], TRILAT_SCALE)
    vocab = []
    for i in range(3):
        for j in range(i + 1, 3):
            (xi, yi), (xj, yj) = st[i], st[j]
            l2 = (xi - xj) ** 2 + (yi - yj) ** 2
            vocab.append(Construction(
                f"cos{i}{j}", "cos", i, 0.0,
                f"(x - ({xi}))*(x - ({xj})) + (y - ({yi}))*(y - ({yj}))"
                f" - ({(d2[i] + d2[j] - l2) / 2})", ("x", "y")))
            vocab += _pins(f"cross{i}{j}_{{}}", "branch", i,
                           f"(x - ({xi}))*(y - ({yj})) - (y - ({yi}))*(x - ({xj}))"
                           " - ({})", ("x", "y"), _heron_sq(d2[i], d2[j], l2))
    return graph, TRILAT_SCALE, vocab


def gen_circles(rng):
    ang = rng.uniform(-math.pi, math.pi)
    R = rng.uniform(7.0, 13.0)
    ax, ay = R * math.cos(ang), R * math.sin(ang)
    rA, rB = rng.uniform(1.0, 3.0), rng.uniform(1.0, 3.0)
    lo, hi = abs(rA - rB), rA + rB
    D = lo + (hi - lo) * rng.uniform(0.15, 0.9)
    b = rng.uniform(-math.pi, math.pi)
    bx, by = ax + D * math.cos(b), ay + D * math.sin(b)
    graph = _g(["x", "y"],
               [("cA", f"(x - ({ax}))**2 + (y - ({ay}))**2 - ({rA * rA})"),
                ("cB", f"(x - ({bx}))**2 + (y - ({by}))**2 - ({rB * rB})")],
               CIRC_SCALE)
    vocab = [Construction(
        "radical", "cos", 1, 0.0,
        f"(x - ({ax}))*({bx - ax}) + (y - ({ay}))*({by - ay})"
        f" - ({(rA * rA + D * D - rB * rB) / 2})", ("x", "y"))]
    vocab += _pins("chord_{}", "branch", 1,
                   f"(x - ({ax}))*({by - ay}) - (y - ({ay}))*({bx - ax}) - ({{}})",
                   ("x", "y"), _heron_sq(rA * rA, D * D, rB * rB))
    return graph, CIRC_SCALE, vocab


def gen_conic_ghost(rng):
    """Coaxial ellipse+circle whose axis-elimination quadratic has exactly one
    real intersection root; the other (ghost, w^2 < 0) root's near-tangency
    region is a spurious LM attractor. Resamples until the one-ghost pattern
    holds — a deterministic function of the seed."""
    while True:
        th = rng.uniform(-math.pi, math.pi)
        ux, uy = math.cos(th), math.sin(th)
        nx, ny = -uy, ux
        R = rng.uniform(12.0, 20.0)
        ca = rng.uniform(-math.pi, math.pi)
        cx, cy = R * math.cos(ca), R * math.sin(ca)
        a = rng.uniform(2.5, 5.0)
        b = a / rng.uniform(3.0, 7.0)
        rc = rng.uniform(0.4, 1.2)
        s = rng.uniform(0.40, 0.95) * (a + rc)
        # eliminate w^2 between the two coaxial equations: A u^2 + B u + C = 0
        A, B, C = 1 - b * b / (a * a), -2 * s, s * s + b * b - rc * rc
        disc = B * B - 4 * A * C
        if disc <= 0:
            continue
        roots = sorted((-B + sig * math.sqrt(disc)) / (2 * A) for sig in (1, -1))
        w2 = [b * b * (1 - u * u / (a * a)) for u in roots]
        valid = [i for i, v in enumerate(w2) if v > 1e-6]
        if len(valid) != 1:
            continue
        U = f"((x - ({cx}))*({ux}) + (y - ({cy}))*({uy}))"
        W = f"((x - ({cx}))*({nx}) + (y - ({cy}))*({ny}))"
        graph = _g(["x", "y"],
                   [("ell", f"{U}**2/({a * a}) + {W}**2/({b * b}) - 1"),
                    ("cir", f"({U} - ({s}))**2 + {W}**2 - ({rc * rc})")],
                   CONIC_SCALE)
        # symmetry-gauge vocabulary, all classical elimination on the givens:
        # the elimination quadratic as a redundant lift, both axis roots as
        # branch pins (the ghost pin is the wrong-sign analogue), and the
        # mirror gauge +-w at the real root.
        vocab = [Construction("axis_quad", "cos", 1, 0.0,
                              f"({A})*{U}**2 + ({B})*{U} + ({C})", ("x", "y"))]
        for i, u in enumerate(roots):
            vocab.append(Construction(f"axis_r{i}", "branch", 1,
                                      1.0 if i else -1.0, f"{U} - ({u})", ("x", "y")))
        wv = math.sqrt(w2[valid[0]])
        vocab += [Construction(f"gauge_{'p' if sgn > 0 else 'm'}", "gauge", 1, sgn,
                               f"{W} - ({sgn * wv})", ("x", "y"))
                  for sgn in (1.0, -1.0)]
        return graph, CONIC_SCALE, vocab


CLASSES = [("ik3r_random", gen_ik3r), ("trilat_far", gen_trilat),
           ("circles_far", gen_circles), ("conic_ghost", gen_conic_ghost)]


def _mcnemar(win, loss):
    """Exact one-sided paired McNemar (as in run_geo_repair.py)."""
    d = win + loss
    if d == 0:
        return 0.5
    return sum(math.comb(d, j) for j in range(win, d + 1)) / (2.0 ** d)


def run_class(cname, gen, base, n):
    t0 = time.time()
    fails = []
    for t in range(n):
        seed = base + t
        graph, scale, vocab = gen(random.Random(seed))
        maxres = residual_fn(graph)
        # 2-stream failure selection: never keep a one-bad-draw failure
        if (lm_solve(graph, maxres, seed, scale)
                or lm_solve(graph, maxres, seed + SALT, scale)):
            continue
        fails.append((seed, graph, scale, vocab, maxres))

    per_cons = {}                  # name -> [flips, tries]  (heterogeneity)
    rows = []
    for seed, graph, scale, vocab, maxres in fails:
        e2e = seed + 3 * SALT      # fresh stream, common across every arm
        worked = {}
        for cons in vocab:
            ok = lm_solve(cons.apply(graph), maxres, e2e, scale)
            worked[cons.name] = ok
            c = per_cons.setdefault(cons.name, [0, 0])
            c[0] += ok
            c[1] += 1
        rows.append({
            "seed": seed, "n_cons": len(vocab), "worked": worked,
            "ceiling": any(worked.values()),
            "restart4": lm_solve(graph, maxres, e2e, scale),
            "restart_matched": lm_solve(graph, maxres, e2e, scale,
                                        k=K_REF * len(vocab)),
        })

    nf = len(rows)
    best_name, best_flips = None, -1
    for name, (f, _tries) in per_cons.items():
        if f > best_flips:
            best_name, best_flips = name, f
    rep = {
        "class": cname, "n": n,
        "fail": rate_cell(nf, n),
        "n_fail": nf,
        "vocab_size": rows[0]["n_cons"] if rows else 0,
        "per_construction": {k: {"flips": v[0], "tries": v[1]}
                             for k, v in sorted(per_cons.items())},
        "best_single_name": best_name,   # picked in-sample; pilot-grade only
        "rows": rows,
        "wall_s": time.time() - t0,
    }
    if nf:
        ceil = sum(r["ceiling"] for r in rows)
        r4 = sum(r["restart4"] for r in rows)
        rm = sum(r["restart_matched"] for r in rows)
        bs = sum(bool(r["worked"].get(best_name)) for r in rows)
        rep.update({
            "ceiling": rate_cell(ceil, nf),
            "best_single": rate_cell(bs, nf),
            "restart4": rate_cell(r4, nf),
            "restart_matched": rate_cell(rm, nf),
            # paired, budget-matched comparisons on the common stream
            "mcnemar_ceiling_vs_matched": _mcnemar(
                sum(r["ceiling"] and not r["restart_matched"] for r in rows),
                sum(r["restart_matched"] and not r["ceiling"] for r in rows)),
            "mcnemar_best_vs_restart4": _mcnemar(
                sum(bool(r["worked"].get(best_name)) and not r["restart4"]
                    for r in rows),
                sum(r["restart4"] and not bool(r["worked"].get(best_name))
                    for r in rows)),
        })
    return rep


def fmt(cell):
    return f"{cell['rate']:.2f} [{cell['ci95'][0]:.2f},{cell['ci95'][1]:.2f}]"


def main():
    ap = argparse.ArgumentParser(description="construction repair on hardened real-system classes")
    ap.add_argument("--n", type=int, default=100, help="instances per class")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="/tmp/pilot_real_repair.json")
    args = ap.parse_args()

    print(f"reference = LM k={K_REF}, acceptance max|r|<{TOL} on original factors; "
          f"2-stream failure selection; CRN grading stream = seed+3*SALT")
    reports = []
    for idx, (cname, gen) in enumerate(CLASSES):
        rep = run_class(cname, gen, args.seed + 40_000 * idx, args.n)
        reports.append(rep)
        print(f"[{cname}] fail={fmt(rep['fail'])} n_fail={rep['n_fail']} "
              f"V={rep['vocab_size']} wall={rep['wall_s']:.0f}s", flush=True)

    hdr = (f"{'class':14} {'fail@LMk4':18} {'n_f':>4} {'ceiling(anyV)':18} "
           f"{'best_single':24} {'restart+4':18} {'restart+4V':18}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for rep in reports:
        if not rep["n_fail"]:
            print(f"{rep['class']:14} {fmt(rep['fail']):18} {0:>4} (no failures)")
            continue
        bs = f"{rep['best_single_name']} {fmt(rep['best_single'])}"
        print(f"{rep['class']:14} {fmt(rep['fail']):18} {rep['n_fail']:>4} "
              f"{fmt(rep['ceiling']):18} {bs:24} {fmt(rep['restart4']):18} "
              f"{fmt(rep['restart_matched']):18}")
    print("\nper-construction flip heterogeneity (flips/tries on the failure pool):")
    for rep in reports:
        if not rep["n_fail"]:
            continue
        cells = "  ".join(f"{k}={v['flips']}/{v['tries']}"
                          for k, v in rep["per_construction"].items())
        print(f"  {rep['class']:14} {cells}")
        print(f"  {'':14} McNemar ceiling>restart+4V p={rep['mcnemar_ceiling_vs_matched']:.4f}  "
              f"best_single>restart+4 p={rep['mcnemar_best_vs_restart4']:.4f}")

    biting = [r["class"] for r in reports
              if r["n_fail"] and r["fail"]["rate"] >= 0.2
              and r["mcnemar_ceiling_vs_matched"] < 0.05]
    print(f"\nclasses with >=20% hard-failure rate AND ceiling significantly above "
          f"the enumeration-budget-matched restart control: {biting or 'NONE'}")

    Path(args.out).write_text(json.dumps({
        "reference_solver": dict(REFERENCE_SOLVER), "tol": TOL,
        "acceptance": "numeric max|r|<tol on ORIGINAL factors; aux factors join the "
                      "least-squares only",
        "config": vars(args), "classes": reports,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
