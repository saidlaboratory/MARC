"""Serialize the P3 structure toys to JSON and log the fixed-structure baseline.

What this does, and why each step exists:

1. Build the 6 hand-authored ``Problem`` objects (3 fixed + 3 augmented) from the
   single source of truth, ``marc.eval.structure_toys``.
2. Serialize every graph to ``marc/data/structure_toys/<id>.json`` using the repo's
   own ``marc.graph.serialize.save_graph`` — so the JSON is guaranteed to round-trip
   through whatever loader the rest of the harness (and Sparsh's eval) uses, rather
   than being hand-written against a guessed schema.
3. Write ``marc/data/structure_toys/gold.json`` (id -> solution + auxiliary-variable
   metadata), since ``save_graph`` stores only the graph, not the solution.
4. Run the *fixed-structure* baseline — ``load_solver("refine")``, i.e.
   ``GradientRefinementSolver`` — on all 6 and check every sample with the real
   ``Checker``. Expectation (this IS the H2 evidence):
       fixed toys      -> solve_rate 0.00  (inconsistent; refine cannot zero energy)
       augmented toys  -> solve_rate 1.00  (consistent; refine finds the unique root)
   The fixed/augmented contrast is the positive control: the toys fail *because* the
   latent node is missing, not for any incidental reason.
5. Print a summary table and write the machine-readable run to
   ``results/p3_structure/structure_toys_baseline.json``.

Note on logging for ticket closure: ``results/`` is gitignored by design, so the
committed evidence is (a) this script + the toys/README, and (b) the printed table
pasted into the PR description. Don't rely on the results JSON being tracked.

Run from the repo root, refine solver (the default; no checkpoint needed):

    python -m scripts.run_structure_toys
    # or: python scripts/run_structure_toys.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from marc.cas.checker import Checker
from marc.eval.solver import load_solver
from marc.eval.structure_toys import (
    AUX_INFO,
    all_structure_toys,
    structure_toys_augmented,
    structure_toys_fixed,
)
from marc.graph.serialize import save_graph

# repo-root-relative output locations
DATA_DIR = Path("marc/data/structure_toys")
RESULTS_DIR = Path("results/p3_structure")
RESULTS_PATH = RESULTS_DIR / "structure_toys_baseline.json"

K = 4  # match the harness default (suites.py); irrelevant to the inconsistent toys
       # but gives refine 4 restarts on the augmented positive controls.


def _solve_rate(accepts: List[List[bool]]) -> float:
    """pass@1: fraction of problems whose FIRST sample the checker accepted."""
    if not accepts:
        return 0.0
    return sum(1 for a in accepts if a and a[0]) / len(accepts)


def _pass_at_k(accepts: List[List[bool]]) -> float:
    """fraction of problems with ANY accepted sample among the k."""
    if not accepts:
        return 0.0
    return sum(1 for a in accepts if any(a)) / len(accepts)


def _run(problems, solver, checker: Checker) -> Dict[str, Any]:
    per_problem: List[Dict[str, Any]] = []
    accepts: List[List[bool]] = []
    for p in problems:
        samples = solver.sample(p, K)
        results = [checker.verify(p.graph, x) for x in samples]
        acc = [r.accepted for r in results]
        accepts.append(acc)
        first = results[0]
        per_problem.append({
            "id": p.id,
            "toy": p.metadata.get("toy"),
            "split": p.metadata.get("split"),
            "n_vars": len(p.graph.variables),
            "accepted_pass1": acc[0] if acc else False,
            "accepted_any": any(acc),
            "first_max_residual": first.max_residual,
            "first_reject_stage": first.stage,
        })
    return {
        "solve_rate": _solve_rate(accepts),
        "pass_at_k": _pass_at_k(accepts),
        "per_problem": per_problem,
    }


def _serialize_graphs() -> List[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for p in all_structure_toys():
        path = DATA_DIR / f"{p.id}.json"
        save_graph(p.graph, str(path))
        paths.append(str(path))
    return paths


def _write_gold() -> str:
    gold: Dict[str, Any] = {}
    for p in all_structure_toys():
        toy = p.metadata.get("toy")
        entry = {
            "solution": p.solution,
            "variables": [v.id for v in p.graph.variables],
            "split": p.metadata.get("split"),
            "description": p.description,
        }
        if p.metadata.get("split") == "structure_toy_augmented" and toy in AUX_INFO:
            entry["auxiliary"] = AUX_INFO[toy]
        gold[p.id] = entry
    path = DATA_DIR / "gold.json"
    path.write_text(json.dumps(gold, indent=2))
    return str(path)


def _print_table(fixed: Dict[str, Any], aug: Dict[str, Any]) -> None:
    by_id = {r["id"]: r for r in fixed["per_problem"] + aug["per_problem"]}
    print("\n=== P3 structure-toys baseline (solver=refine, k=%d) ===" % K)
    print(f"{'id':<18}{'n_vars':>7}{'pass@1':>8}{'pass@k':>8}{'max|resid|':>13}  stage")
    for tid in ["toy1_fixed", "toy1_augmented",
                "toy2_fixed", "toy2_augmented",
                "toy3_fixed", "toy3_augmented"]:
        r = by_id[tid]
        print(f"{tid:<18}{r['n_vars']:>7}"
              f"{('yes' if r['accepted_pass1'] else 'no'):>8}"
              f"{('yes' if r['accepted_any'] else 'no'):>8}"
              f"{r['first_max_residual']:>13.4g}  {r['first_reject_stage'] or '-'}")
    print(f"\nfixed     solve_rate = {fixed['solve_rate']:.2f}   "
          f"(expected 0.00 — inconsistent, latent missing)")
    print(f"augmented solve_rate = {aug['solve_rate']:.2f}   "
          f"(expected 1.00 — latent present, unique solution)")
    ok = fixed["solve_rate"] == 0.0 and aug["solve_rate"] == 1.0
    print("H2 baseline: " + ("PASS — fails on fixed, solves on augmented"
                             if ok else "CHECK — did not match expected contrast"))


def main() -> None:
    checker = Checker()
    solver = load_solver("refine")

    graph_paths = _serialize_graphs()
    gold_path = _write_gold()

    fixed = _run(structure_toys_fixed(), solver, checker)
    aug = _run(structure_toys_augmented(), solver, checker)

    _print_table(fixed, aug)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "solver": getattr(solver, "name", "refine"),
        "k": K,
        "fixed": fixed,
        "augmented": aug,
        "graph_paths": graph_paths,
        "gold_path": gold_path,
    }, indent=2))
    print(f"\nwrote graphs -> {DATA_DIR}/*.json")
    print(f"wrote gold   -> {gold_path}")
    print(f"wrote run    -> {RESULTS_PATH}  (gitignored; paste the table into the PR)")


if __name__ == "__main__":
    main()
