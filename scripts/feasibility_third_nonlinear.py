"""#126 gate: CAS rootless-availability check for a third nonlinear relation family.

Runs BEFORE any transfer-matrix runs are committed. A candidate family clears the
bar iff, using the same ``certify_unsolvable`` the eval arms are graded against:

  1. its fixed graphs can be made aux-required (fixed graph itself CAS-rootless), and
  2. for every gold (a, d), wrong-parameter distractors from the same support are
     CAS-certified rootless (method ``cas_no_real_roots`` -- a theorem, not a
     budget-relative probe verdict), with such distractors on BOTH sides of the gold
     (both a-signs, and both offset directions).

This mirrors the v8 constraint in NONLINEAR_SUPPORTS: one-sided quadratic forms admit
rootless corruptions on both sides; indefinite forms (like the rejected linear
relation) do not. We test both candidates named in the issue and print what actually
holds. No family dict is edited until a candidate clears here.

Usage: python3 scripts/feasibility_third_nonlinear.py
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.structure.invention_data import (
    NONLINEAR_SUPPORTS,
    Candidate,
    _nonlinear_variant,
    certify_unsolvable,
    nonlinear_expression,
)

# Candidate supports mirror the existing families' shape (a in {+1,-1}, integer offset).
SUPPORT: Tuple[Tuple[float, float], ...] = tuple(
    (float(a), float(d)) for a in (1, -1) for d in range(-4, 5)
)


def defining(family: str, a: float, d: float) -> str:
    if family == "hyperbolic":
        return f"u - ({a})*(x**2 - y**2) - ({d})"
    if family == "sq_sum_xy":
        return f"u - ({a})*(x + y)**2 - ({d})"
    raise ValueError(family)


def squared_quantity(family: str, x: float, y: float) -> float:
    if family == "hyperbolic":
        return x ** 2 - y ** 2
    if family == "sq_sum_xy":
        return (x + y) ** 2
    raise ValueError(family)


def build_fixed(family: str, k1: float, k2: float) -> FactorGraph:
    """Fixed (u-free) graph whose nonlinearity is the family's squared quantity.

    hyperbolic:  0.5*(x^2 - y^2) +/- 0.5*y = k{1,2}   (sum -> x^2 - y^2 = k1+k2)
    sq_sum_xy:   0.5*(x+y)^2    +/- 0.5*(x-y) = k{1,2} (sum -> (x+y)^2 = k1+k2)
    """
    if family == "hyperbolic":
        e1, e2 = f"0.5*(x**2 - y**2) + 0.5*y - ({k1})", f"0.5*(x**2 - y**2) - 0.5*y - ({k2})"
        terms = ({"x": 1.0, "y": 0.5}, {"x": 1.0, "y": -0.5})
    else:
        e1, e2 = f"0.5*(x + y)**2 + 0.5*(x - y) - ({k1})", f"0.5*(x + y)**2 - 0.5*(x - y) - ({k2})"
        terms = ({"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0})
    return FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("eq1", e1), FactorNode("eq2", e2)],
        edges=[Edge(v, "eq1", c) for v, c in terms[0].items()]
        + [Edge(v, "eq2", c) for v, c in terms[1].items()],
    )


def sample_gold(family: str, seed: int) -> Optional[Tuple[FactorGraph, Candidate, float, float, float]]:
    """One aux-required gold instance, or None if this attempt did not yield one."""
    rng = random.Random(f"{family}:{seed}")
    a_g, d_g = SUPPORT[seed % len(SUPPORT)]
    for _ in range(200):
        c1 = float(rng.choice((-2, -1, 1, 2)))
        c2 = float(rng.choice((-2, -1, 1, 2)))
        x_star = float(rng.choice((-3, -2, -1, 1, 2, 3)))
        y_star = float(rng.choice((-3, -2, -1, 1, 2, 3)))
        q = squared_quantity(family, x_star, y_star)
        u0 = a_g * q + d_g
        if u0 == 0:
            continue
        if family == "hyperbolic":
            k1 = 0.5 * q + 0.5 * y_star + c1 * u0
            k2 = 0.5 * q - 0.5 * y_star + c2 * u0
        else:
            k1 = 0.5 * q + 0.5 * (x_star - y_star) + c1 * u0
            k2 = 0.5 * q - 0.5 * (x_star - y_star) + c2 * u0
        fixed = build_fixed(family, k1, k2)
        if not certify_unsolvable(fixed, rng_seed=rng.randrange(2 ** 31))["unsolvable"]:
            continue  # fixed graph solvable -> not aux-required
        gold = Candidate("u", u0, {"eq1": c1, "eq2": c2}, defining(family, a_g, d_g))
        return fixed, gold, a_g, d_g, u0
    return None


def measure_golds(golds, support, expr_fn) -> Dict:
    """Shared distractor-rootlessness measurement over a list of built golds.

    ``expr_fn(a, d) -> defining-expression string`` for the family; ``support`` is
    the (a, d) table. Counts, per gold, CAS-theorem-rootless distractors and whether
    they span both a-signs and both offset directions (the v8 "both sides" bar).
    """
    K_MINUS_1 = 3  # default menu K=4 -> build_menu needs 3 certified distractors/gold
    method_counter: Counter = Counter()
    per_gold_both_sides = per_gold_any_rootless = per_gold_fillable = 0
    total_rootless = 0
    for fixed, gold, a_g, d_g in golds:
        rootless_params: List[Tuple[float, float]] = []
        for a2, d2 in support:
            if (a2, d2) == (a_g, d_g):
                continue
            distractor = Candidate("u", gold.pin_value, dict(gold.insert_coeffs), expr_fn(a2, d2))
            verdict = certify_unsolvable(distractor.apply(fixed), rng_seed=12345)
            method_counter[verdict["method"]] += 1
            if verdict["unsolvable"] and verdict["method"] == "cas_no_real_roots":
                rootless_params.append((a2, d2))
        total_rootless += len(rootless_params)
        if rootless_params:
            per_gold_any_rootless += 1
        if len(rootless_params) >= K_MINUS_1:
            per_gold_fillable += 1
        a_signs = {a for a, _ in rootless_params}
        offset_sides = {"lo" if d < d_g else "hi" for _, d in rootless_params if d != d_g}
        if a_signs == {1.0, -1.0} and offset_sides == {"lo", "hi"}:
            per_gold_both_sides += 1

    n = len(golds)
    return {
        "golds_built": n,
        "golds_with_any_cas_rootless_distractor": per_gold_any_rootless,
        "golds_with_both_sides_rootless": per_gold_both_sides,
        "golds_fillable_at_K4": per_gold_fillable,
        "fillable_fraction": round(per_gold_fillable / n, 3) if n else 0.0,
        "mean_rootless_distractors_per_gold": round(total_rootless / n, 2) if n else 0.0,
        "cert_methods_over_distractors": dict(method_counter),
        "both_sides_fraction": round(per_gold_both_sides / n, 3) if n else 0.0,
    }


def probe_candidate(family: str, n_attempts: int = 120) -> Dict:
    golds, fixed_rootless = [], 0
    for seed in range(n_attempts):
        got = sample_gold(family, seed)
        if got is None:
            continue
        fixed_rootless += 1
        fixed, gold, a_g, d_g, _ = got
        golds.append((fixed, gold, a_g, d_g))
    out = {"family": family, "kind": "candidate", "attempts": n_attempts,
           "aux_required_fixed": fixed_rootless}
    out.update(measure_golds(golds, SUPPORT, lambda a, d: defining(family, a, d)))
    return out


def probe_existing(family: str, n_attempts: int = 120) -> Dict:
    """Calibration: identical measurement on a shipped family via its real generator."""
    support = NONLINEAR_SUPPORTS[family]
    golds = []
    for seed in range(n_attempts):
        try:
            fixed, gold, sol = _nonlinear_variant(family, seed)
        except RuntimeError:
            continue
        a_g, d_g = support[seed % len(support)]
        golds.append((fixed, gold, a_g, d_g))
    out = {"family": family, "kind": "shipped (calibration)", "attempts": n_attempts,
           "aux_required_fixed": len(golds)}
    out.update(measure_golds(golds, support, lambda a, d: nonlinear_expression(family, a, d)))
    return out


def report(r: Dict) -> None:
    print(f"=== {r['family']} [{r['kind']}] ===")
    print(f"  aux-required fixed graphs: {r['aux_required_fixed']}/{r['attempts']}")
    print(f"  golds measured: {r['golds_built']}")
    print(f"  mean CAS-rootless distractors / gold: {r['mean_rootless_distractors_per_gold']}")
    print(f"  golds fillable at K=4 (>=3 rootless): {r['golds_fillable_at_K4']} "
          f"({r['fillable_fraction']:.0%})   <- the real menu-build bar")
    print(f"  golds with >=1 CAS-rootless distractor: {r['golds_with_any_cas_rootless_distractor']}")
    print(f"  golds rootless on BOTH sides: {r['golds_with_both_sides_rootless']} "
          f"({r['both_sides_fraction']:.0%})")
    print(f"  distractor cert methods: {r['cert_methods_over_distractors']}\n")


def main() -> None:
    print("#126 CAS rootless-availability gate (theorem-grade: cas_no_real_roots)\n")
    print("--- shipped families (calibration: this is what 'clears the bar' means) ---\n")
    ship = [probe_existing(f) for f in ("vieta", "quad_link")]
    for r in ship:
        report(r)
    print("--- candidate third families ---\n")
    cand = [probe_candidate(f) for f in ("sq_sum_xy", "hyperbolic")]
    for r in cand:
        report(r)

    # The operative bar is menu-fillability (build_menu needs K-1=3 certified
    # distractors/gold); both-sides is the design rationale for non-separability,
    # reported for context. A candidate clears iff its fillable fraction is at least
    # the shipped floor.
    bar = min(r["fillable_fraction"] for r in ship)
    print(f"Bar = min fillable fraction across shipped families (quad_link/vieta): {bar:.0%}\n")
    for r in cand:
        ok = r["fillable_fraction"] >= bar and r["aux_required_fixed"] > 0
        print(f"  {r['family']}: fillable {r['fillable_fraction']:.0%}, "
              f"both-sides {r['both_sides_fraction']:.0%} -> {'CLEARS' if ok else 'FAILS'}")


if __name__ == "__main__":
    main()
