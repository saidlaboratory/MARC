"""A deliberately tiny natural-language -> FactorGraph parser.

CONCEPT.md flags "converting natural language to graphs" as a separate hard
problem (autoformalization), out of MVP scope. This module does **not** attempt
that — it recognizes a small, closed set of sentence templates (linear
sum/difference, bilinear sum/product, and two-distance geometry) and turns a match
into a real :class:`~marc.graph.graph.FactorGraph`. It exists so
``scripts/demo_end_to_end.py``'s "optional NL input" step is a real parser, not a
mock, for the handful of shapes it covers — not a general math-word-problem solver.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from marc.data.geometry import build_triangle_graph
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

_NUM = r"-?\d+"


class NLParseError(ValueError):
    """Raised when ``parse`` can't match any of the known sentence templates."""


def _normalize(text: str) -> str:
    t = f" {text.strip().lower()} "
    for phrase, symbol in (
        (" plus ", " + "),
        (" minus ", " - "),
        (" times ", " * "),
        (" multiplied by ", " * "),
        (" is equal to ", " = "),
        (" equal to ", " = "),
        (" equals ", " = "),
    ):
        t = t.replace(phrase, symbol)
    t = t.replace(",", " ")
    return re.sub(r"\s+", " ", t).strip()


def _sum_diff(text: str) -> Optional[Tuple[FactorGraph, Dict[str, float]]]:
    """"x + y = A and x - y = B" (or the worded equivalent)."""
    m = re.search(rf"x\s*\+\s*y\s*=\s*({_NUM}).*x\s*-\s*y\s*=\s*({_NUM})", text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("eq1", f"x + y - ({a})"), FactorNode("eq2", f"x - y - ({b})")],
        edges=[Edge("x", "eq1", 1), Edge("y", "eq1", 1), Edge("x", "eq2", 1), Edge("y", "eq2", -1)],
    )
    return graph, {"x": (a + b) / 2, "y": (a - b) / 2}


def _sum_product(text: str) -> Optional[Tuple[FactorGraph, Dict[str, float]]]:
    """"x + y = S and x * y = P" (or the worded equivalent) — bilinear, no closed form here."""
    m = re.search(rf"x\s*\+\s*y\s*=\s*({_NUM}).*x\s*\*\s*y\s*=\s*({_NUM})", text)
    if not m:
        return None
    s, p = int(m.group(1)), int(m.group(2))
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("eq1", f"x + y - ({s})"), FactorNode("eq2", f"x*y - ({p})")],
        edges=[Edge("x", "eq1", 1), Edge("y", "eq1", 1), Edge("x", "eq2", 1), Edge("y", "eq2", 1)],
    )
    return graph, {}


def _geometry(text: str) -> Optional[Tuple[FactorGraph, Dict[str, float]]]:
    """"a point at squared distance B from the origin and squared distance A from (C, 0)"."""
    m = re.search(
        rf"(?:squared\s+)?distance\s*({_NUM})\s*from\s*the\s*origin"
        rf".*(?:squared\s+)?distance\s*({_NUM})\s*from\s*\(\s*({_NUM})\s*0\s*\)",
        text,
    )
    if not m:
        return None
    b_sq, a_sq, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return build_triangle_graph(b_sq, a_sq, c), {}


_PATTERNS = (_sum_diff, _sum_product, _geometry)


def parse(text: str) -> Tuple[FactorGraph, Dict[str, float]]:
    """Parse one of the known NL templates into ``(FactorGraph, known_solution)``.

    ``known_solution`` is ``{}`` when the pattern has no closed form (bilinear,
    geometry) — the caller solves it, it isn't looked up.

    Raises:
        NLParseError: if ``text`` doesn't match any known template.
    """
    normalized = _normalize(text)
    for pattern in _PATTERNS:
        result = pattern(normalized)
        if result is not None:
            return result
    raise NLParseError(
        f"could not parse {text!r} into a known graph pattern "
        "(sum+difference, sum+product, or two-distance geometry)"
    )
