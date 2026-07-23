"""Figure for R10: the v0.3 candidate-conditioned repair ranker.

Left panel: the ranker vs its two controls on the three generalization tests that
matter — an unseen linear pattern, balanced nonlinear menus, and transfer to an
unseen nonlinear relation. Right panel: the K=4 checkpoint evaluated zero-shot at
larger menu sizes, where accuracy stays above random while the enumeration it
avoids grows.

Reads the Data Version 8 citable JSONs in results/p_repair/. Run from repo root:
    python3 scripts/plot_repair.py                 # both panels -> fig_repair.pdf
    python3 scripts/plot_repair.py --panel left    # accuracy -> fig_repair_accuracy.pdf (main text)
    python3 scripts/plot_repair.py --panel right   # K-scaling -> fig_repair_kscaling.pdf (appendix)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "p_repair"

# Okabe-Ito, colorblind-safe: ranker = blue, controls = grays.
COLORS = {"ranker": "#0072B2", "candidate-only": "#767676", "random": "#BBBBBB"}
FS = {"tick": 8, "label": 9, "title": 9, "legend": 8}  # shared across panels/figures


def _arm(node, arm):
    n = node[arm]
    n = n.get("invention", n)  # per-family and top-level nest differently
    return n["rate"]


def load_generalization():
    linear = json.loads((RES / "random_support_holdout_shared.json").read_text())
    shared = linear["result"]["per_family"]["aux_required:shared"]
    nonlin = json.loads((RES / "nonlinear_balanced_full.json").read_text())["result"]
    quad = json.loads((RES / "nonlinear_holdout_quad.json").read_text())
    trans = quad["result"]["per_family"]["nonlinear:quad_link"]
    labels = ["Linear\n(unseen pattern)", "Nonlinear\n(balanced)", "Vieta →\nunseen relation"]
    nodes = [shared, nonlin, trans]
    return labels, {
        "ranker": [_arm(n, "full") for n in nodes],
        "candidate-only": [_arm(n, "control") for n in nodes],
        "random": [_arm(n, "random") for n in nodes],
    }


def load_scaling():
    ks, full, rand, calls = [], [], [], []
    for k, f in [(4, "linear_e2e"), (8, "linear_K8_e2e"), (16, "linear_K16_e2e")]:
        r = json.loads((RES / f"{f}.json").read_text())["result"]
        ks.append(k)
        full.append(_arm(r, "full"))
        rand.append(_arm(r, "random"))
        calls.append(r["solve"]["enumeration_cost"]["calls_per_instance"])
    return ks, full, rand, calls


def _accuracy(ax, labels, gen):
    x = range(len(labels))
    w = 0.26
    for i, (name, vals) in enumerate(gen.items()):
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=name, color=COLORS[name])
    ax.axhline(0.25, ls=":", lw=1, color="k", alpha=0.6)
    ax.text(len(labels) - 1.4, 0.265, "K=4 chance", fontsize=FS["tick"] - 1, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=FS["tick"])
    ax.set_ylabel("invention accuracy", fontsize=FS["label"])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FS["legend"], frameon=False)
    ax.set_title("Operator-aware repair beats its controls", fontsize=FS["title"])


def _kscaling(ax, ks, full, rand, calls):
    w = 0.26
    xb = range(len(ks))
    ax.bar([xi - w / 2 for xi in xb], full, w, label="ranker", color=COLORS["ranker"])
    ax.bar([xi + w / 2 for xi in xb], rand, w, label="random", color=COLORS["random"])
    ax.set_xticks(list(xb))
    ax.set_xticklabels([f"K={k}" for k in ks], fontsize=FS["tick"])
    ax.set_ylabel("invention accuracy", fontsize=FS["label"])
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=FS["legend"], frameon=False, loc="upper right")
    ax.set_title("K=4 checkpoint, zero-shot to larger menus", fontsize=FS["title"])
    for xi, c in zip(xb, calls):
        ax.text(xi, 0.02, f"enum\n{c:.1f} calls", fontsize=FS["tick"] - 1.5, ha="center", alpha=0.7)


def main():
    ap = argparse.ArgumentParser(description="R10 repair-ranker figure")
    ap.add_argument("--panel", choices=["left", "right", "both"], default="both",
                    help="left=accuracy (main text), right=K-scaling (appendix)")
    args = ap.parse_args()
    labels, gen = load_generalization()
    ks, full, rand, calls = load_scaling()
    figs = REPO / "paper" / "figures"

    if args.panel == "left":
        fig, ax = plt.subplots(figsize=(4.6, 3.4)); _accuracy(ax, labels, gen)
        out = figs / "fig_repair_accuracy.pdf"
    elif args.panel == "right":
        fig, ax = plt.subplots(figsize=(4.6, 3.4)); _kscaling(ax, ks, full, rand, calls)
        out = figs / "fig_repair_kscaling.pdf"
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        _accuracy(ax[0], labels, gen); _kscaling(ax[1], ks, full, rand, calls)
        out = figs / "fig_repair.pdf"

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
