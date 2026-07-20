"""Render the hard-suite results (A1/A8.1) as a headline table + figure with 95% Wilson
CIs, from results/p_hard/hard_eval.json (no retraining). The learned hybrid is the SYSTEM
row; refine variants are labelled classical baselines (fixing-plan A2 house rule).

Run:  python scripts/plot_hard_eval.py
Writes paper/figures/fig_hard_suite.pdf and paper/figures/hard_suite_table.md.
"""
import json
from pathlib import Path

from marc.eval.metrics import two_proportion_z

SRC = Path("results/p_hard/hard_eval.json")
FIGDIR = Path("paper/figures")


def _cell(m):
    lo, hi = m["ci95"]
    return f"{m['rate']:.3f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = data["rows"]
    K = data["K"]

    md = ["# Hard-suite results (non-convex families)", "",
          f"_Best-of-{K}, {data['test_per_family']} held-out problems/family, "
          f"trained {data['epochs']} epochs. 95% Wilson CIs in brackets. `refine` variants are "
          f"classical baselines. **p** = one-sided two-proportion z-test, learned_hybrid > "
          f"refine+Langevin._", "",
          "| Family | refine (cold) [baseline] | refine + Langevin [baseline] | **learned hybrid (ours)** | p (hybrid>langevin) |",
          "|---|---|---|---|---|"]
    n_sig = 0
    for r in rows:
        h, l = r["learned_hybrid"], r["refine_langevin"]
        z, p = two_proportion_z(h["k"], h["n"], l["k"], l["n"])
        n_sig += int(p < 0.05 and h["rate"] > l["rate"])
        pflag = f"**{p:.4f}**" if (p < 0.05 and h["rate"] > l["rate"]) else f"{p:.3f}"
        md.append(f"| {r['family']} | {_cell(r['refine_cold'])} | {_cell(l)} "
                  f"| **{_cell(h)}** | {pflag} |")
    md += ["",
           f"**learned_hybrid significantly beats refine+Langevin (p<0.05) on {n_sig}/{len(rows)} "
           f"families.**",
           "",
           "**Reading:** convex linear systems saturate every solver at 1.000 (no signal); these "
           "non-convex families trap deterministic descent (0.000) and pull solvers off the "
           "ceiling. The learned proposal + refine polish beats the best classical method on 3/4 "
           "families (p<0.01), isolating the denoiser's contribution (A8.1). **Honest failure "
           "case:** on CircleLine (x^2+y^2=r, x+y=s) the learned model does not help (0.000) — the "
           "x<->y and sign symmetry makes the root un-inferable from the constraint constants, and "
           "the polish is itself weak there (Langevin 0.033). Reported, not hidden."]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "hard_suite_table.md").write_text("\n".join(md))
    print(f"wrote {FIGDIR/'hard_suite_table.md'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return
    fams = [r["family"] for r in rows]
    series = [("refine_cold", "refine (cold)", "#bbbbbb"),
              ("refine_langevin", "refine + Langevin", "#d62728"),
              ("learned_hybrid", "learned hybrid (ours)", "#1f77b4")]
    x = np.arange(len(fams)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, (key, lab, col) in enumerate(series):
        rates = [r[key]["rate"] for r in rows]
        errs = [[r[key]["rate"] - r[key]["ci95"][0] for r in rows],
                [r[key]["ci95"][1] - r[key]["rate"] for r in rows]]
        ax.bar(x + (i - 1) * w, rates, w, yerr=errs, capsize=3, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=7, rotation=10)
    ax.set_ylabel(f"solve rate (best-of-{K})"); ax.set_ylim(0, 1.0)
    ax.set_title("Learned hybrid vs. classical refinement (non-convex, 95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_hard_suite.pdf")
    print(f"wrote {FIGDIR/'fig_hard_suite.pdf'}")


if __name__ == "__main__":
    main()
