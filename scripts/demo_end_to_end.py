#!/usr/bin/env python3
"""End-to-end demo: optional NL input -> graph -> solve -> checker -> print.

Wires together the pieces of the pipeline that exist today for a *new* domain
(geometry): :mod:`marc.nl.parser` (a small closed-set NL->graph parser, not full
autoformalization — see its docstring), :mod:`marc.data.geometry` (the shared
graph builders also used by ``marc/eval/problems.py``'s geometry split), the real
``refine`` energy-gradient solver (:mod:`marc.eval.solver`, swap in ``--solver
learned`` once a checkpoint exists — see ``scripts/train_p2_checkpoints.py`` and
``results/p4_scale/scaling_notes.md`` for the scaled-model training plan), and the
symbolic :class:`~marc.cas.checker.Checker`.

If ``--text`` is omitted, or doesn't match one of the parser's known templates,
the demo falls back to a built-in geometry example so it always runs end-to-end.

Usage:
    python scripts/demo_end_to_end.py
    python scripts/demo_end_to_end.py --text "x plus y equals 5 and x minus y equals 1"
    python scripts/demo_end_to_end.py --text "A point is at squared distance 25 from the origin and squared distance 17 from (4, 0)."
    python scripts/demo_end_to_end.py --solver learned --checkpoint checkpoints/denoiser_stage_a.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.cas.checker import Checker
from marc.eval.solver import load_solver
from marc.nl.parser import NLParseError, parse

DEFAULT_TEXT = (
    "A point is at squared distance 18 from the origin "
    "and squared distance 10 from (4, 0)."
)

# Tuned for the geometry domain's nonconvex quartic energy (see
# marc/eval/problems.py and results/p4_scale/roadmap.md): noise off, smaller
# learning rate, and a much longer polish than the linear-system suites need to
# clear the checker's exact-rational tolerance. --solver refine uses these; a
# --solver learned run ignores them (the diffusion solve() has its own schedule).
GEOMETRY_REFINE_KWARGS = dict(
    steps=1200, lr=0.008, sigma0=0.0, noise=False,
    polish_steps=6000, polish_lr=0.02, init_scale=3.0,
)


class _DemoProblem:
    """Minimal stand-in for :class:`marc.eval.runner.Problem` — just enough shape
    (``.graph``, ``.metadata``) for :class:`~marc.eval.solver.Solver` implementations."""

    def __init__(self, graph):
        self.graph = graph
        self.metadata: dict = {}


def describe_graph(graph) -> str:
    lines = [f"  variables: {[v.id for v in graph.variables]}"]
    for f in graph.factors:
        lines.append(f"  factor {f.id}: {f.expression} = 0")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="MARC end-to-end demo: NL (optional) -> graph -> solve -> checker")
    ap.add_argument("--text", default=None, help="NL sentence; falls back to a built-in geometry example if omitted or unparseable")
    ap.add_argument("--solver", default="refine", choices=["refine", "learned"])
    ap.add_argument("--checkpoint", default=None, help="checkpoint for --solver learned (defaults to $MARC_CKPT)")
    ap.add_argument("--k", type=int, default=12, help="candidates / restarts")
    args = ap.parse_args()

    text = args.text or DEFAULT_TEXT
    if args.text is None:
        print("No --text given; using the built-in geometry example.\n")

    print(f"Input:  {text!r}")
    try:
        graph, _known_solution = parse(text)
    except NLParseError as exc:
        print(f"  -> could not parse that sentence ({exc}); falling back to the built-in geometry example.")
        graph, _known_solution = parse(DEFAULT_TEXT)

    print("Parsed graph (marc.nl.parser -> FactorGraph):")
    print(describe_graph(graph))

    if args.solver == "learned":
        solver = load_solver("learned", checkpoint=args.checkpoint)
    else:
        solver = load_solver("refine", **GEOMETRY_REFINE_KWARGS)

    print(f"\nSolving with --solver {args.solver} (k={args.k}) ...")
    problem = _DemoProblem(graph)
    candidates = [x for x in solver.sample(problem, args.k) if x is not None]

    print("\n" + "-" * 60)
    if not candidates:
        # The learned solver's rollout can diverge to a non-finite energy on a
        # domain its checkpoint never trained on (e.g. the geometry domain's
        # quadratic factors vs. a linear-systems-only checkpoint) and return no
        # candidate at all — a real, reportable failure mode, not a bug to hide.
        print("Solver returned no candidate (every rollout diverged to a "
              "non-finite energy) — likely an out-of-distribution checkpoint "
              "for this domain. See results/p4_scale/roadmap.md.")
        print("-" * 60)
        return

    checker = Checker()
    best = None
    for x in candidates:
        result = checker.verify(graph, x)
        if result.accepted:
            best = (x, result)
            break
        if best is None or result.max_residual < best[1].max_residual:
            best = (x, result)

    x_best, result = best
    names = [v.id for v in graph.variables]
    assignment = ", ".join(f"{n}={v:.6g}" for n, v in zip(names, x_best))

    print(f"Best candidate: {assignment}")
    print(f"Checker: {'ACCEPTED' if result.accepted else 'REJECTED'} (max|r|={result.max_residual:.3e})")
    print("-" * 60)


if __name__ == "__main__":
    main()
