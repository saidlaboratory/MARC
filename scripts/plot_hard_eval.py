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
          f"trained {data['epochs']} epochs. 95% Wilson CIs. `refine` and `random restart` are "
          f"classical baselines; **random restart + polish is the control that isolates the "
          f"learned proposal**._", "",
          "| Family | refine cold | refine+Langevin | **random restart+polish (control)** | learned hybrid | learned>random? |",
          "|---|---|---|---|---|---|"]
    n_lang, n_rand = 0, 0
    for r in rows:
        h, l = r["learned_hybrid"], r["refine_langevin"]
        rnd = r.get("random_restart")
        _, p_l = two_proportion_z(h["k"], h["n"], l["k"], l["n"])
        n_lang += int(p_l < 0.05 and h["rate"] > l["rate"])
        if rnd:
            p_r = r.get("p_learned_gt_random")
            win = "tie/no" if p_r is None or p_r >= 0.05 else "yes"
            n_rand += int(win == "yes")
            md.append(f"| {r['family']} | {r['refine_cold']['rate']:.3f} | {l['rate']:.3f} "
                      f"| **{_cell(rnd)}** | {_cell(h)} | {win} (p={p_r:.2f}) |")
        else:
            md.append(f"| {r['family']} | {_cell(r['refine_cold'])} | {_cell(l)} | n/a | {_cell(h)} | — |")
    md += ["",
           f"**learned_hybrid beats refine+Langevin on {n_lang}/{len(rows)} families (p<0.05), but "
           f"beats the random-restart control on {n_rand}/{len(rows)}.**",
           "",
           "**Honest reading:** the hybrid recipe (a good proposal + energy-descent polish) beats "
           "cold-start Langevin — but a *random* init + the same polish does just as well as the "
           "*learned* proposal on these small-solution families (learned ties on 2, loses on 2). "
           "So the contribution here is the **hybrid recipe**, not the learned denoiser; the "
           "learned proposal's advantage appears only in high dimension where random restart "
           "fails (see the dimension-scaling result). CircleLine is an outright failure (0.000)."]
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
