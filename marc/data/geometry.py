"""Geometry domain graph builders (README §"same template extends to geometry":
coordinates as variables, distance relations as factors).

Shared by :mod:`marc.eval.problems` (random geometry eval splits) and
:mod:`marc.nl.parser` (turning a parsed NL sentence into the same graph shape) so
both stay in exact agreement about what a "geometry problem" is.

Distances are given *squared* (not raw) so every factor stays a polynomial the
symbolic checker can verify exactly — a raw distance would need a ``sqrt``, which
introduces irrational constants the checker's exact-rational gate can't snap to.
"""

from __future__ import annotations

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode


def build_triangle_graph(b_sq: float, a_sq: float, c: float) -> FactorGraph:
    """One unknown point P=(x, y); fixed anchors at the origin and (c, 0).

    ``b_sq`` is the squared distance from P to the origin, ``a_sq`` the squared
    distance from P to (c, 0) — the classic triangle-from-three-side-lengths
    construction (the third side is the anchor-to-anchor distance ``c``).
    """
    return FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[
            FactorNode("eq_origin", f"x**2 + y**2 - ({b_sq})"),
            FactorNode("eq_anchor", f"(x - ({c}))**2 + y**2 - ({a_sq})"),
        ],
        edges=[
            Edge("x", "eq_origin", 1), Edge("y", "eq_origin", 1),
            Edge("x", "eq_anchor", 1), Edge("y", "eq_anchor", 1),
        ],
    )


def build_linked_graph(b1_sq: float, a1_sq: float, link_sq: float, a2_sq: float, c: float) -> FactorGraph:
    """Two unknown points: P1=(x1,y1) as in :func:`build_triangle_graph`, plus a
    second point P2=(x2,y2) linked to P1 by ``link_sq`` and to the anchor (c, 0) by
    ``a2_sq`` — a genuinely more-coupled, 4-variable held-out structure."""
    return FactorGraph(
        variables=[VariableNode("x1"), VariableNode("y1"), VariableNode("x2"), VariableNode("y2")],
        factors=[
            FactorNode("eq_origin", f"x1**2 + y1**2 - ({b1_sq})"),
            FactorNode("eq_anchor1", f"(x1 - ({c}))**2 + y1**2 - ({a1_sq})"),
            FactorNode("eq_link", f"(x2 - x1)**2 + (y2 - y1)**2 - ({link_sq})"),
            FactorNode("eq_anchor2", f"(x2 - ({c}))**2 + y2**2 - ({a2_sq})"),
        ],
        edges=[
            Edge("x1", "eq_origin", 1), Edge("y1", "eq_origin", 1),
            Edge("x1", "eq_anchor1", 1), Edge("y1", "eq_anchor1", 1),
            Edge("x1", "eq_link", -1), Edge("y1", "eq_link", -1),
            Edge("x2", "eq_link", 1), Edge("y2", "eq_link", 1),
            Edge("x2", "eq_anchor2", 1), Edge("y2", "eq_anchor2", 1),
        ],
    )
