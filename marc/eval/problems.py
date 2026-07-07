"""Problem generators for the P1 eval (TECHNICAL_GUIDE §8.3, §11).

Three families, each returning ``Problem`` objects with a *known* exact solution
so the checker is the ground truth:

* :func:`in_distribution` — 2-variable linear systems (the training structure).
  Convex energy; the gradient solver should nail these.
* :func:`held_out_structure` — 3-variable linear systems (more variables/edges, a
  genuine structural shift). Drives the generalization gap (H1 / derive-not-recall).
* :func:`entrapment_suite` — nonconvex single-variable residuals with a real root
  *and* a spurious E > 0 local minimum behind an energy barrier. Plain gradient
  descent stalls in the spurious basin a sizeable fraction of the time; injected
  noise escapes. This is the RQ2 noise-on/off probe.

Splits are tagged in ``metadata["split"]`` so reporting can document them per §11.
"""

from __future__ import annotations

import random
from typing import List

from marc.data.geometry import build_linked_graph, build_triangle_graph
from marc.eval.runner import Problem
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode


def in_distribution(n: int = 25, seed: int = 0) -> List[Problem]:
    """2-variable linear systems:  x + y = a,  x - y = b  (solution ((a+b)/2, (a-b)/2))."""
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(n):
        a = rng.randint(-9, 9)
        b = rng.randint(-9, 9)
        graph = FactorGraph(
            variables=[VariableNode("x"), VariableNode("y")],
            factors=[
                FactorNode("eq1", f"x + y - ({a})"),
                FactorNode("eq2", f"x - y - ({b})"),
            ],
            edges=[
                Edge("x", "eq1", 1), Edge("y", "eq1", 1),
                Edge("x", "eq2", 1), Edge("y", "eq2", -1),
            ],
        )
        problems.append(
            Problem(
                id=f"id_{i:03d}",
                graph=graph,
                solution=[(a + b) / 2, (a - b) / 2],
                description=f"x+y={a}, x-y={b}",
                metadata={"split": "in_distribution", "n_vars": 2},
            )
        )
    return problems


def held_out_structure(n: int = 25, seed: int = 1) -> List[Problem]:
    """3-variable linear systems — held-out structure (more vars/edges than train).

    x + y + z = a,  x - y = b,  y - z = c. Solved exactly by back-substitution:
    z = (a - b - 2c)/3, y = z + c, x = y + b.
    """
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(n):
        a = rng.randint(-9, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-9, 9)
        z = (a - b - 2 * c) / 3
        y = z + c
        x = y + b
        graph = FactorGraph(
            variables=[VariableNode("x"), VariableNode("y"), VariableNode("z")],
            factors=[
                FactorNode("eq1", f"x + y + z - ({a})"),
                FactorNode("eq2", f"x - y - ({b})"),
                FactorNode("eq3", f"y - z - ({c})"),
            ],
            edges=[
                Edge("x", "eq1", 1), Edge("y", "eq1", 1), Edge("z", "eq1", 1),
                Edge("x", "eq2", 1), Edge("y", "eq2", -1),
                Edge("y", "eq3", 1), Edge("z", "eq3", -1),
            ],
        )
        problems.append(
            Problem(
                id=f"ho_{i:03d}",
                graph=graph,
                solution=[x, y, z],
                description=f"x+y+z={a}, x-y={b}, y-z={c}",
                metadata={"split": "held_out_structure", "n_vars": 3},
            )
        )
    return problems


def linear_system(n_vars: int, n: int = 25, seed: int = 0) -> List[Problem]:
    """Uniquely-solvable ``n_vars``-variable linear systems (length-extrapolation axis).

    "Global sum + consecutive differences" structure:

        sum_j x_j = sum_j s_j        (one global-sum factor)
        x_{i-1} - x_i = s_{i-1} - s_i   (i = 1 .. n_vars-1)

    The unique solution is ``s = (s0, ..., s_{n_vars-1})``, so the checker is the
    ground truth at any length. This is deliberately the *same family* the rest of
    the harness trains/tests on: ``n_vars = 2`` reproduces :func:`in_distribution`
    (x+y, x-y) and ``n_vars = 3`` reproduces :func:`held_out_structure` (sum + two
    differences). Larger ``n_vars`` extend that structure to longer systems the model
    never saw — the length-extrapolation probe (§11).

    The family stays well-conditioned with length, so the energy-gradient baseline
    solves every length to checker precision; a model that *memorised* short systems
    is the one expected to fall off as length grows.
    """
    if n_vars < 2:
        raise ValueError("n_vars must be >= 2")
    # offset the seed by length so each bucket draws independent constants
    rng = random.Random(seed + 1009 * n_vars)
    problems: List[Problem] = []
    for i in range(n):
        s = [rng.randint(-9, 9) for _ in range(n_vars)]
        variables = [VariableNode(f"x{j}") for j in range(n_vars)]

        total = sum(s)
        sum_expr = " + ".join(f"x{j}" for j in range(n_vars))
        factors = [FactorNode("eq0", f"{sum_expr} - ({total})")]
        edges = [Edge(f"x{j}", "eq0", 1) for j in range(n_vars)]
        for j in range(1, n_vars):
            fid = f"eq{j}"
            d = s[j - 1] - s[j]
            factors.append(FactorNode(fid, f"x{j - 1} - x{j} - ({d})"))
            edges.append(Edge(f"x{j - 1}", fid, 1))
            edges.append(Edge(f"x{j}", fid, -1))

        graph = FactorGraph(variables=variables, factors=factors, edges=edges)
        problems.append(
            Problem(
                id=f"len{n_vars}_{i:03d}",
                graph=graph,
                solution=[float(v) for v in s],
                description=f"sum + consecutive differences, {n_vars} vars",
                metadata={"split": f"length_{n_vars}", "n_vars": n_vars},
            )
        )
    return problems


def geometry_in_distribution(n: int = 25, seed: int = 3) -> List[Problem]:
    """Geometry domain (README §"same template extends to geometry"): coordinates as
    variables, distance relations as factors. One unknown point P=(x, y); two fixed
    anchors at the origin and (c, 0). Given the (integer, squared to stay polynomial
    and avoid irrational constants) distances from P to each anchor, solve for P —
    the classic triangle-from-three-side-lengths construction.
    """
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(n):
        # Small integer ranges keep the two circles' squared-distance constants
        # modest (see scripts/run_geometry_eval.py's tuning notes) — with the
        # generic energy-gradient refine() this is a well-conditioned but
        # genuinely nonconvex (quartic-energy) domain, unlike the convex linear
        # systems the other suites use, so wide constant ranges make plain
        # gradient descent's outer basin unreliable from generic random inits.
        c = rng.randint(3, 5)
        x = rng.choice([v for v in range(-4, 5) if v != 0])
        y = rng.choice([v for v in range(-4, 5) if v != 0])
        b_sq = x ** 2 + y ** 2          # squared distance P -> origin
        a_sq = (x - c) ** 2 + y ** 2    # squared distance P -> (c, 0)
        graph = build_triangle_graph(b_sq, a_sq, c)
        problems.append(
            Problem(
                id=f"geo_{i:03d}",
                graph=graph,
                solution=[float(x), float(y)],
                description=f"point at squared-distance {b_sq} from origin, {a_sq} from ({c},0)",
                metadata={"split": "geometry_in_distribution", "n_vars": 2, "anchor": c},
            )
        )
    return problems


def geometry_held_out(n: int = 25, seed: int = 4) -> List[Problem]:
    """Held-out geometry structure: a second unknown point P2 chained off the first
    (its distance to P1 is a factor, not just to the fixed anchors) — a 4-variable,
    genuinely more-coupled shape than :func:`geometry_in_distribution`'s single
    triangle, mirroring how :func:`held_out_structure` chains an extra variable onto
    :func:`in_distribution`.
    """
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(n):
        c = rng.randint(3, 5)
        x1 = rng.choice([v for v in range(-4, 5) if v != 0])
        y1 = rng.choice([v for v in range(-4, 5) if v != 0])
        x2 = rng.choice([v for v in range(-4, 5) if v != 0])
        y2 = rng.choice([v for v in range(-4, 5) if v != 0])

        b1_sq = x1 ** 2 + y1 ** 2
        a1_sq = (x1 - c) ** 2 + y1 ** 2
        link_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2   # P2 -> P1, couples the two points
        a2_sq = (x2 - c) ** 2 + y2 ** 2
        graph = build_linked_graph(b1_sq, a1_sq, link_sq, a2_sq, c)
        problems.append(
            Problem(
                id=f"geoho_{i:03d}",
                graph=graph,
                solution=[float(x1), float(y1), float(x2), float(y2)],
                description=f"two-point chain off anchor ({c},0)",
                metadata={"split": "geometry_held_out", "n_vars": 4, "anchor": c},
            )
        )
    return problems


def entrapment_suite(n: int = 50, seed: int = 7) -> List[Problem]:
    """Nonconvex residuals that trap gradient descent (RQ2 probe).

    Residual r(x) = (x - R) * ((x - m)^2 + h), with h > 0. The quadratic factor has
    no real root, so x = R is the *only* solution (E = 0 there). But near x = m the
    quadratic bottoms out at h, giving |r| a shallow local minimum r(m) = (m - R)*h
    behind an energy barrier — a locally-consistent-but-globally-wrong fixed point.
    ``h`` tunes the spurious-basin depth; R - m tunes the barrier width.

    Each start (``metadata['init']``) sits inside the spurious basin, so plain
    gradient descent (noise off) is trapped by construction; the noise-on arm must
    cross the barrier to reach R. Both arms see the identical, seeded start point.
    """
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(n):
        # round every constant to the precision printed into the expression so the
        # stored solution is an *exact* root of the polynomial the checker sees
        R = round(rng.uniform(1.5, 2.2), 6)
        m = round(rng.uniform(-0.2, 0.2), 6)
        h = round(rng.uniform(0.1, 0.3), 6)
        # r(x) = (x - R)((x - m)^2 + h); a cubic the checker/CAS see as a polynomial
        expr = f"(x - ({R})) * ((x - ({m}))**2 + ({h}))"
        init = round(m + rng.uniform(-0.2, 0.2), 6)  # inside the spurious basin
        graph = FactorGraph(
            variables=[VariableNode("x")],
            factors=[FactorNode("eq1", expr)],
            edges=[Edge("x", "eq1", 1)],
        )
        problems.append(
            Problem(
                id=f"trap_{i:03d}",
                graph=graph,
                solution=[R],
                description=f"cubic root at x={R:.3f}, spurious basin near x={m:.3f}",
                metadata={"split": "entrapment", "n_vars": 1, "init": [init]},
            )
        )
    return problems
