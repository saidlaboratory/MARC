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
