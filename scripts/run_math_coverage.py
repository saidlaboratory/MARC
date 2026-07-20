"""MATH benchmark coverage (reality check).

Honest question this answers: of real competition-math problems (MATH-500 sample),
how many can MARC's autoformalization pipeline even *ingest* into a FactorGraph, and
of those, how many does the solver get right? This is a **coverage** measurement, not a
claim that MARC solves MATH — MARC targets the constraint-shaped slice, and the
formalizer (NL -> graph) is the bottleneck (CONCEPT.md defers general autoformalization).

For every problem: try `marc.nl.parser.parse`; if it parses, solve the graph with the
classical refine solver and compare to the gold answer. Report:
  * parser coverage (fraction that formalize at all)
  * solve accuracy on the covered subset
  * breakdown by subject and difficulty level

Run:  python scripts/run_math_coverage.py
Writes results/p_math/coverage.json and prints a summary.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from marc.nl.parser import parse, NLParseError
from marc.cas.checker import Checker
from marc.refine.iterative import refine

DATA = Path("marc/data/math_benchmark/math500_sample.jsonl")


def load():
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def _num(s: str):
    """Best-effort numeric value of a gold answer string (int/float/simple fraction)."""
    s = s.strip().strip("$").replace(" ", "")
    if re.fullmatch(r"-?\d+", s):
        return float(s)
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def try_solve(problem: dict):
    """Returns (parsed: bool, correct: bool|None). correct is None if unparsed."""
    try:
        graph, known = parse(problem["problem"])
    except NLParseError:
        return False, None
    chk = Checker()
    # solve: use the known solution if the template gives one, else refine
    if known:
        x = [known[v.id] for v in graph.variables]
    else:
        best = None
        for s in range(16):
            tr = refine(graph, [0.0] * len(graph.variables), noise=True, seed=s)
            if chk.verify(graph, tr.x).accepted:
                best = tr.x
                break
        x = best
    if x is None:
        return True, False
    gold = _num(problem["answer"])
    if gold is None:
        return True, None  # parsed & solved, but gold not numerically comparable
    return True, any(abs(v - gold) < 1e-6 for v in x)


def main() -> None:
    problems = load()
    n = len(problems)
    parsed = solved = 0
    by_type = Counter()
    by_level = Counter()
    for p in problems:
        by_type[p["type"]] += 1
        by_level[p["level"]] += 1
        ok, correct = try_solve(p)
        if ok:
            parsed += 1
            if correct:
                solved += 1

    print(f"MATH-500 sample: {n} problems")
    print(f"parser coverage : {parsed}/{n} = {parsed/n:.3f}")
    denom = parsed if parsed else 1
    print(f"solve accuracy on covered : {solved}/{parsed} = {solved/denom:.3f}")
    print("\nby subject:")
    for t, c in by_type.most_common():
        print(f"  {t:24s} {c:3d}")
    print("by level:")
    for lv in sorted(by_level):
        print(f"  level {lv}: {by_level[lv]}")

    out = {
        "n": n, "parser_coverage": parsed / n,
        "solve_accuracy_on_covered": solved / denom,
        "parsed": parsed, "solved": solved,
        "by_type": dict(by_type), "by_level": dict(by_level),
        "note": "Coverage of the constraint-shaped slice; formalization (NL->graph) is the "
                "bottleneck, not the solver. MARC does not target general MATH.",
    }
    out_dir = Path("results/p_math")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir/'coverage.json'}")


if __name__ == "__main__":
    main()
