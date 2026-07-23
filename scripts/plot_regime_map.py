"""Regime map: the two-condition law on one axis pair.

X: measured log-q(n) slope (reachability collapse). Y: solution structure
(separable / coupled bands). Every measured family is a point colored by the
measured learned-vs-random outcome, so the law -- learning wins iff reachability
collapses AND the solution is per-variable separable -- is readable as "the win
quadrant contains exactly the winning families".

Slopes for the three core families come from the citable loglin fits in
results/p_crossover/crossover_theory.json. The R27 families
(results/p_scaling/crossover_families.json) report best-of-8 rates only, so
their slopes are inverted through the law itself: q = 1-(1-P)^(1/K) on the LM
arm (the arm with unsaturated rows at every n; the paper's point is that LM
collapses along the same v^n curve). Outcomes are read off p_learned_gt_random.

Run from repo root:
    python3 scripts/plot_regime_map.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

DIVIDER = -0.45  # illustrative: measured slopes sit at -0.13 vs <= -0.77
SEP_LINE = 0.5

C_WIN, C_TIE, C_NA = "#4338ca", "#d97706", "#9ca3af"


def fit_slope(ns, qs):
    xs = [float(n) for n in ns]
    ys = [math.log(q) for q in qs]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def r27_families():
    d = json.loads((RES / "p_scaling" / "crossover_families.json").read_text())
    K = d["K"]
    out = {}
    for name, rows in d["families"].items():
        # ponytail: LM-arm law inversion, not direct q -- R27 measured only best-of-8
        usable = [r for r in rows if 0 < r["lm"]["k"] < r["lm"]["n"]]
        slope = fit_slope([r["n"] for r in usable],
                          [1 - (1 - r["lm"]["rate"]) ** (1 / K) for r in usable])
        win = min(r["p_learned_gt_random"] for r in rows) < 0.05
        out[name] = (slope, win)
    return out


def core_slopes():
    d = json.loads((RES / "p_crossover" / "crossover_theory.json").read_text())
    return {k: d["families"][k]["loglin"]["b"] for k in ("indep", "coupled", "geometry")}


def outcome_checks():
    scaling = json.loads((RES / "p_scaling" / "scaling.json").read_text())
    coupled = json.loads((RES / "p_coupled" / "coupled.json").read_text())
    chains = json.loads((RES / "p_geometry" / "pointchain_learned.json").read_text())
    return (min(r["p_learned_gt_random"] for r in scaling["rows"]) < 0.05,
            min(r["p_learned_gt_random"] for r in coupled["rows"]) < 0.05,
            min(r["p_learned_gt_random"] for r in chains["rows"]) < 0.05)


def main():
    core = core_slopes()
    r27 = r27_families()
    indep_wins, coupled_wins, chains_win = outcome_checks()
    # labels below are hardcoded; these asserts fail loudly if the JSONs move under them
    assert indep_wins and not coupled_wins and not chains_win
    assert r27["baseline"][1] and r27["wide_roots"][1] and not r27["double_well"][1]

    # (name, slope, y, outcome, label, label offset (pts), ha)
    pts = [
        ("bundled traps (R15)", core["indep"], 1.08, "win",
         "bundled traps (R15)\n0.975 vs 0.075 at $n{=}3$", (6, 4), "left"),
        ("R27 baseline", r27["baseline"][0], 0.70, "win",
         "R27 baseline\n1.000 vs 0.000 at $n{=}6$", (6, -10), "left"),
        ("R27 wide roots", r27["wide_roots"][0], 1.35, "win",
         "R27 wide roots\n0.225 vs 0.000 at $n{=}4$", (4, -24), "left"),
        ("R27 double well", r27["double_well"][0], 1.60, "tie",
         "R27 double well: tie\n(denoiser under-fit; capacity, not regime)", (8, -6), "left"),
        ("chained bilinear (R7)", core["coupled"], 0.08, "tie",
         "chained bilinear (R7)\n0/5 significant wins", (0, 8), "center"),
        ("point chains (R25)", core["geometry"], 0.08, "tie",
         "point chains (R25)\nlearned ties random,\ncollapses with it", (8, 8), "left"),
    ]
    for name, slope, y, oc, *_ in pts:
        if oc == "win":
            assert slope < DIVIDER and y > SEP_LINE, name

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set_xlim(-1.62, 0.12)
    ax.set_ylim(-0.78, 1.78)

    # quadrant structure
    ax.axvline(DIVIDER, color="#9ca3af", lw=0.8, ls="--")
    ax.axhline(SEP_LINE, color="#9ca3af", lw=0.8, ls="--")
    ax.fill_between([-1.62, DIVIDER], SEP_LINE, 1.78, color=C_WIN, alpha=0.07, lw=0)
    cell = dict(fontsize=7.5, style="italic", color="#374151", alpha=0.9)
    ax.text(-1.59, 1.70, "collapse + separable:\nlearning wins", va="top", **cell)
    ax.text(-0.41, 1.70, "no collapse:\nrandom survives, nothing to win", va="top", **cell)
    ax.text(-1.59, -0.70, "collapse + coupled: tie\n(the law's falsifiable cell --- held)",
            va="bottom", **cell)
    ax.text(-0.41, -0.70, "no collapse + coupled:\nnothing to amortize, tie",
            va="bottom", **cell)

    style = {"win": (C_WIN, "o", True), "tie": (C_TIE, "s", True), "na": (C_NA, "^", False)}
    for name, slope, y, oc, lab, (dx, dy), ha in pts:
        col, mk, filled = style[oc]
        ax.plot([slope], [y], mk, color=col, mfc=col if filled else "white",
                ms=7, mew=1.4, zorder=5)
        ax.annotate(lab, (slope, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7, ha=ha, color="#111827", zorder=6)

    # R26 real systems: classical-arms only, slope not measured -> nominal abscissa
    ax.plot([-0.06], [-0.35], "^", color=C_NA, mfc="white", ms=7, mew=1.4, zorder=5)
    ax.annotate("8 real systems (R26)\nLM 8/8; learned n.a.\n(slope not measured)",
                (-0.06, -0.35), textcoords="offset points", xytext=(-8, 0),
                fontsize=7, ha="right", va="center", color="#111827", zorder=6)

    # R28: the structural-decision counterpart of the geometry tie
    ax.annotate("TODO R28 (in flight): same pruned chains,\nlearning relocated to the "
                "structural decision",
                (core["geometry"], 0.02), textcoords="offset points", xytext=(-30, -52),
                fontsize=7, ha="center", color="#6b7280",
                arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.8, ls=":"))

    ax.set_xlabel("measured $\\log q(n)$ slope"
                  "$\\quad(\\leftarrow$ reachability collapses)", fontsize=9)
    ax.set_yticks([0.0, 1.1])
    ax.set_yticklabels(["coupled", "separable"], fontsize=9, rotation=90, va="center")
    ax.set_ylabel("solution structure", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Learned value proposals win iff reachability collapses "
                 "and solutions are per-variable separable", fontsize=9)

    handles = [Line2D([], [], marker="o", ls="", color=C_WIN, ms=6, label="learned wins"),
               Line2D([], [], marker="s", ls="", color=C_TIE, ms=6, label="learned ties random"),
               Line2D([], [], marker="^", ls="", color=C_NA, mfc="white", ms=6,
                      label="learned arm n.a.")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left",
              bbox_to_anchor=(0.01, 0.12))

    fig.tight_layout()
    out = REPO / "paper" / "figures" / "fig_regime_map.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    for name, slope, y, oc, *_ in pts:
        print(f"  {name}: slope {slope:.2f}, {oc}")


if __name__ == "__main__":
    main()
