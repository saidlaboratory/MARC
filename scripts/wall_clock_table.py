#!/usr/bin/env python3
"""The deployment-facing wall-clock receipt for the paper's cost claims.

Every arm's per-instance wall time is already measured inside the committed
result JSONs at one boundary on one machine (the R28 run grades all arms in a
single pass; R10 records the ranker forward pass). This re-emits them as one
table so "the ranker's single call is cheaper" carries a number, not an
assertion. Reads only committed JSONs — no re-run.

  python3 scripts/wall_clock_table.py        # markdown to stdout
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    geo = json.loads((ROOT / "results/p_geo_repair/analysis_v3.json").read_text())["cost"]
    nl = json.loads((ROOT / "results/p_repair/nonlinear_balanced_full_paired.json").read_text())
    fwd = nl["result"]["inference_timing"]["full_forward_s_per_instance"]

    print("## Wall-clock receipts (one machine, per-instance)\n")
    print("R28 geometry repair — every arm graded in one pass (`analysis_v3.json`), so the "
          "restarts and wall time are directly comparable:\n")
    print("| arm | restarts/instance | wall ms/instance | vs ranker |")
    print("|---|---:|---:|---:|")
    base = geo["ranker"]["wall_s_per_instance"]["mean"]
    for arm in ("ranker", "recipe_only", "restart_control", "restart_plus16",
                "restart_plus32", "probe", "enumeration"):
        c = geo[arm]
        ms = 1000 * c["wall_s_per_instance"]["mean"]
        k = c["restarts_per_instance"]["mean"]
        rel = c["wall_s_per_instance"]["mean"] / base
        print(f"| {arm} | {k:.1f} | {ms:.0f} | {rel:.2f}x |")
    print(f"\nThis prices the arms: a single augmented solve ({1000*base:.0f} ms) is "
          f"{geo['probe']['wall_s_per_instance']['mean']/base:.1f}x cheaper than the probe and "
          f"{geo['enumeration']['wall_s_per_instance']['mean']/base:.0f}x cheaper than "
          f"enumeration. On R28 that buys nothing — the ranker's pick ties the prior and "
          f"loses to the matched-budget restart control on accuracy (a negative). The cost "
          f"win is R10:")
    print(f"\nR10 nonlinear repair — the learned selection is a single forward pass of "
          f"{fwd*1000:.2f} ms/instance over the candidate graphs (deterministic featurization "
          f"cached) plus one reference solve, matching the enumeration ceiling at a fraction "
          f"of its 2.62 solves per instance — cost win and accuracy at once (R10).")


if __name__ == "__main__":
    main()
