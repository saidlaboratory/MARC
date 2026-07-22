"""Figure for R10: the v0.3 candidate-conditioned repair ranker.

Left panel: the ranker vs its two controls on the three generalization tests that
matter — an unseen linear pattern, balanced nonlinear menus, and transfer to an
unseen nonlinear relation. Right panel: the K=4 checkpoint evaluated zero-shot at
larger menu sizes, where accuracy stays above random while the enumeration it
avoids grows.

Reads the Data Version 8 citable JSONs in results/p_repair/. Run from repo root:
    python3 scripts/plot_repair.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "p_repair"


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


def main():
    labels, gen = load_generalization()
    ks, full, rand, calls = load_scaling()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

    x = range(len(labels))
    w = 0.26
    colors = {"ranker": "#4338ca", "candidate-only": "#9ca3af", "random": "#d1d5db"}
    for i, (name, vals) in enumerate(gen.items()):
        ax[0].bar([xi + (i - 1) * w for xi in x], vals, w, label=name, color=colors[name])
    ax[0].axhline(0.25, ls=":", lw=1, color="k", alpha=0.6)
    ax[0].text(len(labels) - 1.4, 0.265, "K=4 chance", fontsize=7, alpha=0.7)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("invention accuracy")
    ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("Operator-aware repair beats its controls", fontsize=9)

    xb = range(len(ks))
    ax[1].bar([xi - w / 2 for xi in xb], full, w, label="ranker", color=colors["ranker"])
    ax[1].bar([xi + w / 2 for xi in xb], rand, w, label="random", color=colors["random"])
    ax[1].set_xticks(list(xb))
    ax[1].set_xticklabels([f"K={k}" for k in ks])
    ax[1].set_ylabel("invention accuracy")
    ax[1].set_ylim(0, 0.7)
    ax[1].legend(fontsize=8, frameon=False, loc="upper right")
    ax[1].set_title("K=4 checkpoint, zero-shot to larger menus", fontsize=9)
    for xi, c in zip(xb, calls):
        ax[1].text(xi, 0.02, f"enum\n{c:.1f} calls", fontsize=6.5, ha="center", alpha=0.7)

    fig.tight_layout()
    out = REPO / "paper" / "figures" / "fig_repair.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
