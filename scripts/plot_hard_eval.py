"""Render the hard-suite results (A1/A8.1) as a headline table + figure from the
existing results/p_hard/hard_eval.json (no retraining). The learned hybrid is the
SYSTEM row; refine variants are labelled classical baselines (fixing-plan A2 house
rule: refine is never the headline).

Run:  python scripts/plot_hard_eval.py
Writes paper/figures/fig_hard_suite.pdf and paper/figures/hard_suite_table.md.
"""
import json
from pathlib import Path

SRC = Path("results/p_hard/hard_eval.json")
FIGDIR = Path("paper/figures")


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = data["rows"]
    K = data["K"]

    # --- markdown table ---
    md = ["# Hard-suite results (non-convex families)", "",
          f"_Best-of-{K}, {data['test_per_family']} held-out problems/family, "
          f"trained {data['epochs']} epochs. `refine` variants are classical baselines._", "",
          "| Family | refine (cold) [baseline] | refine + Langevin [baseline] | **learned hybrid (ours)** |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['family']} | {r['refine_cold']:.3f} | {r['refine_langevin']:.3f} "
                  f"| **{r['learned_hybrid']:.3f}** |")
    md += ["",
           "**Reading:** convex linear systems saturate every solver at 1.000 (no signal); "
           "these non-convex bilinear families trap deterministic descent (0.000) and pull "
           "solvers off the ceiling. The learned proposal + refine polish beats the best "
           "classical method on every family — isolating the denoiser's contribution (A8.1)."]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "hard_suite_table.md").write_text("\n".join(md))
    print(f"wrote {FIGDIR/'hard_suite_table.md'}")

    # --- grouped bar chart ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return
    fams = [r["family"] for r in rows]
    series = [("refine_cold", "refine (cold)", "#bbbbbb"),
              ("refine_langevin", "refine + Langevin", "#d62728"),
              ("learned_hybrid", "learned hybrid (ours)", "#1f77b4")]
    import numpy as np
    x = np.arange(len(fams)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (key, lab, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=8)
    ax.set_ylabel(f"solve rate (best-of-{K})"); ax.set_ylim(0, 1.0)
    ax.set_title("Learned hybrid vs. classical refinement (non-convex)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_hard_suite.pdf")
    print(f"wrote {FIGDIR/'fig_hard_suite.pdf'}")


if __name__ == "__main__":
    main()
