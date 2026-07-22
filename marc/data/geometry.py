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

import random
from dataclasses import dataclass
from typing import Dict, Tuple

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


def build_point_chain_graph(link_sqs, anchor_sqs, origin_sq: float, c: float) -> FactorGraph:
    """A chain of ``k`` unknown points P_1..P_k (2k variables). P_1 is anchored to the
    origin (``origin_sq``) and to (c,0) (``anchor_sqs[0]``); each subsequent P_{i+1} is
    linked to P_i by ``link_sqs[i-1]`` and anchored to (c,0) by ``anchor_sqs[i]``.

    Generalises :func:`build_triangle_graph` (k=1) and :func:`build_linked_graph`
    (k=2) to arbitrary length, giving a *coupled* geometry family that scales in
    dimension — used to test the factorization law (R9) on a real-ish domain. Each
    point contributes exactly two constraints (one coupling, one anchor), so the
    system stays square and the chain propagates: basins do NOT factorize per
    variable, so the law predicts a flat single-start reachability q(n)."""
    k = len(anchor_sqs)
    vs, fs, es = [], [], []
    for i in range(k):
        vs += [VariableNode(f"x{i}"), VariableNode(f"y{i}")]
    # P_0: anchored to origin and to (c,0)
    fs.append(FactorNode("eq_origin", f"x0**2 + y0**2 - ({origin_sq})"))
    fs.append(FactorNode("eq_anchor0", f"(x0 - ({c}))**2 + y0**2 - ({anchor_sqs[0]})"))
    es += [Edge("x0", "eq_origin", 1), Edge("y0", "eq_origin", 1),
           Edge("x0", "eq_anchor0", 1), Edge("y0", "eq_anchor0", 1)]
    for i in range(1, k):
        fs.append(FactorNode(f"eq_link{i}",
                             f"(x{i} - x{i-1})**2 + (y{i} - y{i-1})**2 - ({link_sqs[i-1]})"))
        fs.append(FactorNode(f"eq_anchor{i}", f"(x{i} - ({c}))**2 + y{i}**2 - ({anchor_sqs[i]})"))
        es += [Edge(f"x{i-1}", f"eq_link{i}", -1), Edge(f"y{i-1}", f"eq_link{i}", -1),
               Edge(f"x{i}", f"eq_link{i}", 1), Edge(f"y{i}", f"eq_link{i}", 1),
               Edge(f"x{i}", f"eq_anchor{i}", 1), Edge(f"y{i}", f"eq_anchor{i}", 1)]
    return FactorGraph(variables=vs, factors=fs, edges=es)


def make_point_chain(k: int, rng):
    """Integer-coordinate coupled geometry chain of ``k`` points (2k variables).
    Returns (graph, solution). Coordinates are small nonzero integers so the
    checker's exact-rational gate accepts, matching the geometry eval splits."""
    c = rng.randint(3, 5)
    pts = [(rng.choice([v for v in range(-4, 5) if v != 0]),
            rng.choice([v for v in range(-4, 5) if v != 0])) for _ in range(k)]
    origin_sq = pts[0][0] ** 2 + pts[0][1] ** 2
    anchor_sqs = [(x - c) ** 2 + y ** 2 for (x, y) in pts]
    link_sqs = [(pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2
                for i in range(1, k)]
    g = build_point_chain_graph(link_sqs, anchor_sqs, origin_sq, c)
    sol = [float(v) for p in pts for v in p]
    return g, sol


def make_pruned_chain(k: int, rng, n_extra: int | None = None):
    """DMDGP-style instance: a k-point chain plus ``n_extra`` long-range squared
    distances between non-adjacent points (default ceil(k/2)).

    The plain chain has ~2^(k-1) branch-variant real solutions (every consecutive
    circle-circle intersection is a free reflection), so "solve the system" is
    easy and the exact checker's preferred solution is just one sheet among many.
    The long-range distances prune that tree — most reflection strings become
    infeasible, wrong branches turn into nonzero-residual local minima, and
    multistart search genuinely fails. This is the discretizable distance
    geometry setting (sparse exact distances -> conformation), built the way
    DMDGP benchmarks are: from a known configuration.

    Returns (graph, solution, givens) where ``givens`` holds exactly the data a
    solver is allowed to see: {"c", "origin_sq", "anchor_sqs", "link_sqs",
    "extra": [(i, j, d_sq), ...]}. Construction vocabularies must derive from
    givens only — never from the solution.
    """
    if n_extra is None:
        n_extra = (k + 1) // 2
    c = rng.randint(3, 5)
    pts = [(rng.choice([v for v in range(-4, 5) if v != 0]),
            rng.choice([v for v in range(-4, 5) if v != 0])) for _ in range(k)]
    origin_sq = pts[0][0] ** 2 + pts[0][1] ** 2
    anchor_sqs = [(x - c) ** 2 + y ** 2 for (x, y) in pts]
    link_sqs = [(pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2
                for i in range(1, k)]
    g = build_point_chain_graph(link_sqs, anchor_sqs, origin_sq, c)
    pairs = [(i, j) for i in range(k) for j in range(i + 2, k)]
    rng.shuffle(pairs)
    extra = []
    vs = list(g.variables)
    fs = list(g.factors)
    es = list(g.edges)
    for (i, j) in pairs[:n_extra]:
        d_sq = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
        fid = f"eq_long{i}_{j}"
        fs.append(FactorNode(fid, f"(x{i} - x{j})**2 + (y{i} - y{j})**2 - ({d_sq})"))
        es += [Edge(f"x{i}", fid, 1), Edge(f"y{i}", fid, 1),
               Edge(f"x{j}", fid, -1), Edge(f"y{j}", fid, -1)]
        extra.append((i, j, d_sq))
    graph = FactorGraph(variables=vs, factors=fs, edges=es)
    sol = [float(v) for p in pts for v in p]
    givens = {"c": float(c), "origin_sq": float(origin_sq),
              "anchor_sqs": [float(v) for v in anchor_sqs],
              "link_sqs": [float(v) for v in link_sqs], "extra": extra}
    return graph, sol, givens


def build_point_chain_graph_3d(link_sqs, anchor2_sqs, anchor3_sqs, origin_sq, c, d):
    """3D (DMDGP) point chain: k unknown points P_0..P_{k-1} (3k variables), each
    determined by THREE squared-distance spheres so the intersection is two points
    (a binary reflection branch across the anchor plane) exactly like molecular
    distance geometry.

    Three fixed, non-collinear anchors span the z=0 plane: A1=origin, A2=(c,0,0),
    A3=(0,d,0). P_0 is pinned to all three (origin_sq, anchor2_sqs[0],
    anchor3_sqs[0]); each P_i (i>=1) is linked to P_{i-1} (link_sqs[i-1]) and pinned
    to A2, A3 (anchor2_sqs[i], anchor3_sqs[i]) -- three constraints, so the system
    is square and every point is a sphere-sphere-sphere intersection with a binary
    branch. Squared distances only, so every factor is an exact polynomial."""
    k = len(anchor2_sqs)
    vs, fs, es = [], [], []
    for i in range(k):
        vs += [VariableNode(f"x{i}"), VariableNode(f"y{i}"), VariableNode(f"z{i}")]
    fs.append(FactorNode("eq_origin", f"x0**2 + y0**2 + z0**2 - ({origin_sq})"))
    fs.append(FactorNode("eq_a2_0", f"(x0 - ({c}))**2 + y0**2 + z0**2 - ({anchor2_sqs[0]})"))
    fs.append(FactorNode("eq_a3_0", f"x0**2 + (y0 - ({d}))**2 + z0**2 - ({anchor3_sqs[0]})"))
    es += [Edge("x0", "eq_origin", 1), Edge("y0", "eq_origin", 1), Edge("z0", "eq_origin", 1),
           Edge("x0", "eq_a2_0", 1), Edge("y0", "eq_a2_0", 1), Edge("z0", "eq_a2_0", 1),
           Edge("x0", "eq_a3_0", 1), Edge("y0", "eq_a3_0", 1), Edge("z0", "eq_a3_0", 1)]
    for i in range(1, k):
        fs.append(FactorNode(f"eq_link{i}",
                             f"(x{i} - x{i-1})**2 + (y{i} - y{i-1})**2 + (z{i} - z{i-1})**2 "
                             f"- ({link_sqs[i-1]})"))
        fs.append(FactorNode(f"eq_a2_{i}", f"(x{i} - ({c}))**2 + y{i}**2 + z{i}**2 - ({anchor2_sqs[i]})"))
        fs.append(FactorNode(f"eq_a3_{i}", f"x{i}**2 + (y{i} - ({d}))**2 + z{i}**2 - ({anchor3_sqs[i]})"))
        es += [Edge(f"x{i-1}", f"eq_link{i}", -1), Edge(f"y{i-1}", f"eq_link{i}", -1),
               Edge(f"z{i-1}", f"eq_link{i}", -1),
               Edge(f"x{i}", f"eq_link{i}", 1), Edge(f"y{i}", f"eq_link{i}", 1),
               Edge(f"z{i}", f"eq_link{i}", 1),
               Edge(f"x{i}", f"eq_a2_{i}", 1), Edge(f"y{i}", f"eq_a2_{i}", 1),
               Edge(f"z{i}", f"eq_a2_{i}", 1),
               Edge(f"x{i}", f"eq_a3_{i}", 1), Edge(f"y{i}", f"eq_a3_{i}", 1),
               Edge(f"z{i}", f"eq_a3_{i}", 1)]
    return FactorGraph(variables=vs, factors=fs, edges=es)


def make_pruned_chain_3d(k: int, rng, n_extra: int | None = None):
    """3D DMDGP-style instance: a k-point 3D chain (:func:`build_point_chain_graph_3d`)
    plus ``n_extra`` long-range squared distances between non-adjacent points
    (default ceil(k/2)), which prune the ~2^k reflection tree the way molecular
    DMDGP benchmarks do -- most reflection strings become infeasible and multistart
    genuinely fails.

    Returns (graph, solution, givens) with ``givens`` holding exactly what a solver
    may see: {"c", "d", "origin_sq", "anchor2_sqs", "anchor3_sqs", "link_sqs",
    "extra": [(i, j, d_sq), ...]}. Vocabularies derive from givens only, never the
    solution. Integer coordinates keep every squared distance an exact integer so
    the checker's exact-rational gate accepts the planted config and its z-mirror."""
    if n_extra is None:
        n_extra = (k + 1) // 2
    c = rng.randint(3, 5)
    d = rng.randint(3, 5)
    nz = [v for v in range(-4, 5) if v != 0]
    # z strictly nonzero so each point sits off the anchor plane (a real branch)
    pts = [(rng.choice(nz), rng.choice(nz), rng.choice(nz)) for _ in range(k)]
    origin_sq = pts[0][0] ** 2 + pts[0][1] ** 2 + pts[0][2] ** 2
    anchor2_sqs = [(x - c) ** 2 + y ** 2 + z ** 2 for (x, y, z) in pts]
    anchor3_sqs = [x ** 2 + (y - d) ** 2 + z ** 2 for (x, y, z) in pts]
    link_sqs = [sum((pts[i][t] - pts[i - 1][t]) ** 2 for t in range(3))
                for i in range(1, k)]
    g = build_point_chain_graph_3d(link_sqs, anchor2_sqs, anchor3_sqs, origin_sq, c, d)
    pairs = [(i, j) for i in range(k) for j in range(i + 2, k)]
    rng.shuffle(pairs)
    vs, fs, es = list(g.variables), list(g.factors), list(g.edges)
    extra = []
    for (i, j) in pairs[:n_extra]:
        d_sq = sum((pts[i][t] - pts[j][t]) ** 2 for t in range(3))
        fid = f"eq_long{i}_{j}"
        fs.append(FactorNode(fid, f"(x{i} - x{j})**2 + (y{i} - y{j})**2 + (z{i} - z{j})**2 - ({d_sq})"))
        es += [Edge(f"x{i}", fid, 1), Edge(f"y{i}", fid, 1), Edge(f"z{i}", fid, 1),
               Edge(f"x{j}", fid, -1), Edge(f"y{j}", fid, -1), Edge(f"z{j}", fid, -1)]
        extra.append((i, j, d_sq))
    graph = FactorGraph(variables=vs, factors=fs, edges=es)
    sol = [float(v) for p in pts for v in p]
    givens = {"c": float(c), "d": float(d), "origin_sq": float(origin_sq),
              "anchor2_sqs": [float(v) for v in anchor2_sqs],
              "anchor3_sqs": [float(v) for v in anchor3_sqs],
              "link_sqs": [float(v) for v in link_sqs], "extra": extra}
    return graph, sol, givens


@dataclass
class PointChainTemplate:
    """Point-chain geometry as a generator template: k points, 2k variables.

    Wraps :func:`make_point_chain` into the (graph, {var: value}) contract the
    trainer's templates use, so the geometry denoiser can train on the same
    family the R9 crossover law flagged as learning-favorable."""

    k: int = 2
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"PointChain{self.k}"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        graph, sol = make_point_chain(self.k, random.Random(seed))
        # make_point_chain's flat solution follows variable order (x0,y0,x1,y1,...)
        return graph, {v.id: s for v, s in zip(graph.variables, sol)}


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
