"""A predictive law for when a learned proposal beats random search (the crossover).

This script turns the empirical R5/R7 crossover into a *mechanistic, falsifiable* law
and validates it with (nearly) zero free parameters. It measures a single geometric
quantity per family and predicts the whole random-restart-vs-dimension curve from it.

------------------------------------------------------------------------------------
The law
------------------------------------------------------------------------------------
All best-of-K methods share one polish operator ``marc.refine.iterative.refine`` and one
``Checker`` gate. For a single random start, let

    q(n) = P[ one random start + polish lands in an accepting basin ]   (the *reachability*).

Best-of-K restart is then, exactly (K i.i.d. starts):

    P_random(n; K) = 1 - (1 - q(n))^K.                                    (binomial, tautological)

The *scientific* content is how q(n) scales, and that is set by whether the acceptance
basins **factorize across variables**:

  * INDEPENDENT traps  (each factor touches one variable; polish is coordinate-separable):
        a start solves iff every coordinate independently lands in its root basin, so
            q(n) = v^n,     v := q(1) = single-coordinate basin fraction.
        => q decays GEOMETRICALLY; log q(n) is linear in n with slope log v.
        => random restart needs ~ v^{-n} starts and collapses; a learned proposal that
           reproduces each variable's marginal root stays ~flat and *must* win for large n.

  * COUPLED system  (chained bilinear; polish couples neighbours):
        the solution is a joint object, the polish propagates along the chain, so the
        reachable set does NOT shrink as v^n. We predict q_coupled(n) stays ~FLAT.
        => random restart does not collapse, and a learned proposal has nothing to
           amortize: it ties random, and a classical joint solver (Levenberg-Marquardt)
           dominates. (This is exactly the R7 negative.)

So one measured number, v = q(1), plus the factorization test, predicts:
  - the entire P_random(n;K) curve for the independent family (parameter-free),
  - that the coupled family has no collapse (falsifiable: q_coupled flat),
  - the crossover dimension n* where a flat learned proposal overtakes random restart:
        n* = ceil( log(1 - (1 - p_L)^{1/K}) / log v ),   p_L = learned solve ceiling.

------------------------------------------------------------------------------------
What this run does
------------------------------------------------------------------------------------
1. Measures q(n) for BOTH families by single-start (K=1) polish over many fresh instances
   (Wilson CIs), using the *identical* generators / refine / Checker as R5 and R7.
2. Tests factorization: fits log q(n) ~ a + b n; reports slope b (=log v) and R^2.
   Independent should be linear with b<0; coupled should be ~flat (b~0).
3. Predicts P_random(n;K) = 1-(1-q(n))^K and compares to the observed random-restart
   rates in results/p_scaling/scaling.json and results/p_coupled/coupled.json.
4. Predicts the crossover n* from v and the learned ceiling p_L, compares to observed.
5. Writes results/p_crossover/crossover_theory.json and paper/figures/fig_crossover_theory.pdf.

Run:  python scripts/run_crossover_theory.py [--trials 300] [--K 8]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from marc.cas.checker import Checker
from marc.data.coupled import make_chain
from marc.eval.metrics import wilson_interval
from marc.refine.iterative import refine
from scripts.run_dimension_scaling import make_problem, DECIMALS

# Dimensions matched to the R5 (independent) and R7 (coupled) experiments.
NS_INDEP = [1, 2, 3, 4, 6]
NS_COUPLED = [2, 3, 4, 6, 8]
INDEP_START = 8.0     # random restart draws x ~ U[-8, 8]   (run_dimension_scaling)
COUPLED_START = 4.0   # random restart draws x ~ U[-4, 4]   (run_coupled_eval)


def _accept_indep(chk: Checker, g, x) -> bool:
    return chk.verify(g, [round(v, DECIMALS) for v in x]).accepted


def _accept_coupled(chk: Checker, g, x) -> bool:
    return chk.verify(g, x).accepted


def _gen(family, n, rng):
    if family == "indep":
        g, sol, _ = make_problem(n, rng)
        return g, sol, _accept_indep
    g, sol = make_chain(n, rng)
    return g, sol, _accept_coupled


def single_start_q(family: str, n: int, trials: int, span: float, seed0: int):
    """Empirical q(n): fraction of ONE random start + polish that lands in an accepting
    basin, over `trials` fresh instances (one start each). This is the K=1 reachability
    that the best-of-K law is built from. noise=False (pure descent), matching the
    random-restart control in both experiments."""
    chk = Checker()
    ok = 0
    for j in range(trials):
        rng = random.Random(seed0 + 7919 * j)
        g, sol, accept = _gen(family, n, rng)
        x0 = [rng.uniform(-span, span) for _ in range(n)]
        ok += accept(chk, g, refine(g, x0, noise=False, seed=0).x)
    return ok, trials


def bestofk_random(family: str, n: int, instances: int, K: int, span: float, seed0: int):
    """Directly measure best-of-K random restart under the SAME conditions as q(n), so the
    predicted-vs-observed comparison is apples-to-apples (same seeds/N, no cross-JSON noise).
    Each instance gets K independent uniform starts + polish; solved if ANY accepts. This
    reproduces the random-restart control of run_dimension_scaling / run_coupled_eval."""
    chk = Checker()
    ok = 0
    for j in range(instances):
        rng = random.Random(seed0 + 4099 * j)
        g, sol, accept = _gen(family, n, rng)
        solved = False
        for _ in range(K):
            x0 = [rng.uniform(-span, span) for _ in range(n)]
            if accept(chk, g, refine(g, x0, noise=False, seed=0).x):
                solved = True
                break
        ok += int(solved)
    return ok, instances


def _loglin_fit(ns, qs):
    """Least-squares fit of log q ~ a + b n over points with q>0. Returns (a, b, r2, used)."""
    pts = [(n, math.log(q)) for n, q in zip(ns, qs) if q > 0]
    if len(pts) < 2:
        return None
    xm = sum(p[0] for p in pts) / len(pts)
    ym = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - xm) ** 2 for p in pts)
    sxy = sum((p[0] - xm) * (p[1] - ym) for p in pts)
    b = sxy / sxx
    a = ym - b * xm
    ss_tot = sum((p[1] - ym) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2, [p[0] for p in pts]


def best_of_k(q: float, K: int) -> float:
    return 1 - (1 - q) ** K


def load_observed(path: str, key: str):
    """Return {n: rate} for a baseline column from an existing results JSON."""
    d = json.loads(Path(path).read_text())
    out = {}
    for r in d.get("rows", []):
        cell = r.get(key)
        if isinstance(cell, dict) and "rate" in cell:
            out[r["n"]] = cell["rate"]
    return out


def measure_family(family: str, ns, span: float, trials: int, K: int, seed0: int):
    rows = []
    print(f"\n[{family}] single-start reachability q(n) + best-of-{K} random restart "
          f"(trials={trials}, start=U[-{span},{span}])")
    for n in ns:
        k, t = single_start_q(family, n, trials, span, seed0 + 101 * n)
        q = k / t
        bk, bt = bestofk_random(family, n, trials, K, span, seed0 + 202 * n)
        exp_starts = (1.0 / q) if q > 0 else float("inf")
        rows.append({"n": n, "q": q, "k": k, "trials": t, "ci95": wilson_interval(k, t),
                     "expected_starts": exp_starts,
                     "P_random_meas": bk / bt, "P_random_meas_ci95": wilson_interval(bk, bt)})
        print(f"  n={n:>2}  q={q:6.3f} ({k}/{t})  E[starts]={exp_starts:7.1f}  "
              f"random@{K}={bk/bt:6.3f} ({bk}/{bt})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictive crossover law: measure v, predict the curve")
    ap.add_argument("--trials", type=int, default=300, help="fresh instances per n for q(n)")
    ap.add_argument("--K", type=int, default=8, help="best-of-K budget (matches R5/R7)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    indep = measure_family("indep", NS_INDEP, INDEP_START, args.trials, args.K, args.seed)
    coupled = measure_family("coupled", NS_COUPLED, COUPLED_START, args.trials, args.K, args.seed + 55555)

    # --- factorization test -----------------------------------------------------------
    fi = _loglin_fit([r["n"] for r in indep], [r["q"] for r in indep])
    fc = _loglin_fit([r["n"] for r in coupled], [r["q"] for r in coupled])
    v = math.exp(fi[0] + fi[1]) if fi else float("nan")   # implied per-var fraction q(1)=e^{a+b}
    v_direct = next((r["q"] for r in indep if r["n"] == 1), None)

    print("\n=== factorization test (log q ~ a + b*n) ===")
    print(f"  indep:   slope b={fi[1]:+.4f}  (=> v=e^b={math.exp(fi[1]):.3f})  R^2={fi[2]:.3f}")
    print(f"  coupled: slope b={fc[1]:+.4f}  R^2={fc[2]:.3f}   (flat b~0 => basins do NOT factorize)")
    print(f"  measured v = q_indep(1) = {v_direct:.3f}")

    # --- prediction of best-of-K random restart, compared to the SELF-MEASURED curve --
    # P_pred(q): the binomial best-of-K from the instance-averaged q(n).
    # P_pred(v^n): the *parameter-free* prediction using only v=q(1) and the factorization
    #              law q(n)=v^n (independent family only; it is the wrong model for coupled).
    learned_indep = load_observed("results/p_scaling/scaling.json", "learned_x0")

    def attach_pred(rows):
        for r in rows:
            r["P_random_pred_q"] = best_of_k(r["q"], args.K)              # from measured q(n)
            r["P_random_pred_vn"] = best_of_k(v_direct ** r["n"], args.K) if v_direct else None
        return rows

    indep = attach_pred(indep)
    coupled = attach_pred(coupled)

    print("\n=== best-of-K random restart: predicted vs self-measured (same conditions) ===")
    print(f"{'fam':>8} {'n':>3} {'q':>7} {'pred 1-(1-q)^K':>15} {'pred 1-(1-v^n)^K':>17} {'measured':>9}")
    for tag, rows in (("indep", indep), ("coupled", coupled)):
        for r in rows:
            pv = f"{r['P_random_pred_vn']:.3f}" if r["P_random_pred_vn"] is not None else "  -  "
            print(f"{tag:>8} {r['n']:>3} {r['q']:>7.3f} {r['P_random_pred_q']:>15.3f} "
                  f"{pv:>17} {r['P_random_meas']:>9.3f}")

    # MAE of the parameter-free v^n prediction against the self-measured random curve (indep).
    mae_vn = [abs(r["P_random_pred_vn"] - r["P_random_meas"])
              for r in indep if r["P_random_pred_vn"] is not None]
    mae_vn = sum(mae_vn) / len(mae_vn) if mae_vn else None

    print("\n=== expected restarts to first success  E[starts]=1/q(n) ===")
    print("  (the budget random search needs; ~v^-n explodes iff basins factorize)")
    for tag, rows in (("indep", indep), ("coupled", coupled)):
        es = "  ".join(f"n{r['n']}:{r['expected_starts']:.0f}" if r['expected_starts'] != float('inf')
                       else f"n{r['n']}:inf" for r in rows)
        print(f"  {tag:>8}: {es}")

    # --- predicted crossover dimension -----------------------------------------------
    # learned ceiling p_L: median learned rate at the low dimensions where it is flat.
    flat_learned = [learned_indep[n] for n in (1, 2, 3) if n in learned_indep]
    p_L = sorted(flat_learned)[len(flat_learned) // 2] if flat_learned else 0.95
    # n* : smallest n with P_random(n;K) < p_L, under the v^n law.
    n_star = None
    if 0 < v_direct < 1:
        thresh = 1 - (1 - p_L) ** (1 / args.K)   # per-start q at which best-of-K = p_L
        n_star = math.ceil(math.log(thresh) / math.log(v_direct)) if thresh > 0 else None
    # observed crossover: first n where learned exceeds the self-measured random curve.
    meas_random = {r["n"]: r["P_random_meas"] for r in indep}
    obs_cross = next((n for n in NS_INDEP
                      if learned_indep.get(n, 0) > meas_random.get(n, 1)), None)

    print("\n=== crossover prediction (independent family) ===")
    print(f"  learned ceiling p_L={p_L:.3f}   v={v_direct:.3f}   K={args.K}")
    print(f"  predicted crossover n* = {n_star}    observed crossover = {obs_cross}")
    print(f"  parameter-free v^n MAE on random-restart curve = {mae_vn:.3f}" if mae_vn is not None else "")

    payload = {
        "trials": args.trials, "K": args.K,
        "families": {
            "indep": {"ns": NS_INDEP, "start_span": INDEP_START, "rows": indep,
                      "loglin": {"a": fi[0], "b": fi[1], "r2": fi[2], "v_from_slope": math.exp(fi[1])}},
            "coupled": {"ns": NS_COUPLED, "start_span": COUPLED_START, "rows": coupled,
                        "loglin": {"a": fc[0], "b": fc[1], "r2": fc[2]}},
        },
        "v_direct": v_direct,
        "p_L": p_L,
        "crossover_pred": n_star, "crossover_obs": obs_cross,
        "random_restart_vn_mae": mae_vn,
        "law": "P_random(n;K)=1-(1-q(n))^K ; q_indep(n)=v^n (factorizes) ; q_coupled(n)~const (does not)",
    }
    out = Path("results/p_crossover"); out.mkdir(parents=True, exist_ok=True)
    (out / "crossover_theory.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out/'crossover_theory.json'}")
    _plot(indep, coupled, v_direct, args.K, n_star, learned_indep)


def _plot(indep, coupled, v, K, n_star, learned_indep) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

    # (a) factorization test: log q vs n
    for rows, col, lab in ((indep, "#1f77b4", "independent traps"),
                           (coupled, "#d62728", "coupled chain")):
        ns = [r["n"] for r in rows]
        qs = [max(r["q"], 1e-3) for r in rows]
        ax[0].plot(ns, qs, "o-", color=col, label=lab, linewidth=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dimension n")
    ax[0].set_ylabel("single-start reachability q(n)")
    ax[0].set_title("(a) basins factorize? q(n)=v$^n$ is a line")
    ax[0].legend(fontsize=8)

    # (b) predicted vs observed random restart + learned + crossover
    ns = [r["n"] for r in indep]
    ax[1].plot(ns, [r["P_random_pred_vn"] for r in indep], "--", color="#9467bd",
               label="random: predicted 1-(1-v$^n$)$^K$", linewidth=2)
    ax[1].plot(ns, [r["P_random_meas"] for r in indep], "v", color="#9467bd",
               label="random: measured", markersize=8)
    ax[1].plot(ns, [learned_indep.get(n) for n in ns], "D-", color="#1f77b4",
               label="learned: observed", linewidth=2)
    if n_star is not None:
        ax[1].axvline(n_star, color="#888", ls=":", label=f"predicted crossover n*={n_star}")
    ax[1].set_xlabel("dimension n")
    ax[1].set_ylabel("solve rate (best-of-K)")
    ax[1].set_ylim(-0.03, 1.05)
    ax[1].set_title("(b) parameter-free prediction of the crossover")
    ax[1].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    d = Path("paper/figures"); d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig_crossover_theory.pdf")
    print(f"wrote {d/'fig_crossover_theory.pdf'}")


if __name__ == "__main__":
    main()
