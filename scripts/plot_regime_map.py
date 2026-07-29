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
    # Labels are short because the figure is placed at \columnwidth: the per-family rates
    # live in the caption and the main text, not on the canvas.
    pts = [
        ("bundled traps (R15)", core["indep"], 1.08, "win",
         "bundled traps (R15)", (6, 0), "left"),
        ("R27 baseline", r27["baseline"][0], 0.72, "win",
         "R27 baseline", (6, 0), "left"),
        ("R27 wide roots", r27["wide_roots"][0], 1.40, "win",
         "R27 wide roots", (6, 0), "left"),
        ("R27 double well", r27["double_well"][0], 1.72, "tie",
         "R27 double well: tie", (6, 0), "left"),
        ("chained bilinear (R7)", core["coupled"], 0.08, "tie",
         "chained bilinear (R7)", (0, 8), "center"),
        ("point chains (R25)", core["geometry"], 0.08, "tie",
         "point chains (R25)", (-7, 0), "right"),
    ]
    for name, slope, y, oc, *_ in pts:
        if oc == "win":
            assert slope < DIVIDER and y > SEP_LINE, name

    # Sized for a single \columnwidth slot (~3.3 in) so the figure is printed at ~1:1 and
    # every label keeps its stated point size. Drawing it at two-column width and shrinking
    # it into a column is what made the annotations unreadable.
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.set_xlim(-1.62, 0.42)
    ax.set_ylim(-0.78, 2.15)

    # quadrant structure
    ax.axvline(DIVIDER, color="#9ca3af", lw=0.8, ls="--")
    ax.axhline(SEP_LINE, color="#9ca3af", lw=0.8, ls="--")
    ax.fill_between([-1.62, DIVIDER], SEP_LINE, 2.15, color=C_WIN, alpha=0.07, lw=0)
    cell = dict(fontsize=6, style="italic", color="#374151", alpha=0.9)
    ax.text(-1.59, 2.10, "collapse + separable:\nlearning wins", va="top", **cell)
    ax.text(-0.41, 2.10, "no collapse:\nnothing to win", va="top", **cell)
    ax.text(-1.59, -0.74, "collapse + coupled: tie\n(falsifiable cell --- held)",
            va="bottom", **cell)
    ax.text(-0.41, -0.74, "no collapse + coupled:\ntie", va="bottom", **cell)

    style = {"win": (C_WIN, "o", True), "tie": (C_TIE, "s", True), "na": (C_NA, "^", False)}
    for name, slope, y, oc, lab, (dx, dy), ha in pts:
        col, mk, filled = style[oc]
        ax.plot([slope], [y], mk, color=col, mfc=col if filled else "white",
                ms=5, mew=1.2, zorder=5)
        ax.annotate(lab, (slope, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6, ha=ha, va="center", color="#111827", zorder=6)

    # R26 real systems: classical-arms only, slope not measured -> nominal abscissa
    ax.plot([-0.06], [-0.22], "^", color=C_NA, mfc="white", ms=5, mew=1.2, zorder=5)
    ax.annotate("8 real systems (R26)\nslope not measured", (-0.06, -0.22),
                textcoords="offset points", xytext=(-7, 0),
                fontsize=6, ha="right", va="center", color="#111827", zorder=6)

    ax.set_xlabel("measured $\\log q(n)$ slope"
                  "$\\quad(\\leftarrow$ reachability collapses)", fontsize=7)
    ax.set_yticks([0.0, 1.1])
    ax.set_yticklabels(["coupled", "separable"], fontsize=7, rotation=90, va="center")
    ax.set_ylabel("solution structure", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0])

    handles = [Line2D([], [], marker="o", ls="", color=C_WIN, ms=4.5, label="learned wins"),
               Line2D([], [], marker="s", ls="", color=C_TIE, ms=4.5, label="ties random"),
               Line2D([], [], marker="^", ls="", color=C_NA, mfc="white", ms=4.5,
                      label="no learned arm")]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.80), handletextpad=0.4, labelspacing=0.25)

    fig.tight_layout()
    out = REPO / "paper" / "tex" / "figures" / "fig_regime_map.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    for name, slope, y, oc, *_ in pts:
        print(f"  {name}: slope {slope:.2f}, {oc}")


if __name__ == "__main__":
    main()
