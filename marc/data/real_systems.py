"""A small suite of *named, standard* nonlinear systems from real domains.

Motivation (external validity): every other family in this study is procedurally
generated. These are recognized test problems from robotics, positioning, chemistry,
classic optimization, and the computer-algebra benchmark literature, encoded once as
factor graphs so the same solver battery and the same reachability measurement apply.
Their real roots are generally irrational, so acceptance here is a numeric residual
tolerance ($\\max_j |r_j(x)| < $ tol), the standard fair criterion for comparing
numerical solvers, rather than MARC's exact-rational checker (which the synthetic
families use because they are constructed to have rational solutions).

Each entry is (name, domain, FactorGraph, init_scale, note). Every system has at least
one real solution; `verify_all()` (and the unit test) confirm a solver finds one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode


@dataclass
class RealSystem:
    name: str
    domain: str
    graph: FactorGraph
    init_scale: float
    note: str


def _g(varnames, factors, init_scale) -> FactorGraph:
    """factors: list of (factor_id, expression). Edges connect every variable that
    appears in an expression to that factor (coefficient 1; the CAS reads the
    expression, so the coefficient only labels connectivity for the graph view)."""
    import sympy as sp

    vs = [VariableNode(v) for v in varnames]
    fs, es = [], []
    for fid, expr in factors:
        fs.append(FactorNode(fid, expr))
        present = {s.name for s in sp.sympify(expr).free_symbols}
        for v in varnames:
            if v in present:
                es.append(Edge(v, fid, 1))
    return FactorGraph(variables=vs, factors=fs, edges=es)


def real_systems() -> List[RealSystem]:
    S: List[RealSystem] = []

    # 1. Two-circle intersection (planar geometry): the classic construction.
    S.append(RealSystem(
        "circle_intersection", "geometry",
        _g(["x", "y"], [("c1", "x**2 + y**2 - 4"), ("c2", "(x - 3)**2 + y**2 - 4")], 3.0),
        3.0, "unit-radius circles centered at (0,0) and (3,0); two real intersections"))

    # 2. Conic-line intersection: ellipse meets a line.
    S.append(RealSystem(
        "conic_line", "geometry",
        _g(["x", "y"], [("ell", "x**2 + 2*y**2 - 3"), ("lin", "x - y")], 3.0),
        3.0, "ellipse x^2+2y^2=3 intersect line x=y; roots (1,1),(-1,-1)"))

    # 3. Trilateration / planar positioning (GPS-style): three range circles about
    #    anchors, consistent at (1,1). A real localization problem, overdetermined.
    S.append(RealSystem(
        "trilateration", "positioning",
        _g(["x", "y"], [("a0", "x**2 + y**2 - 2"),
                        ("a1", "(x - 4)**2 + y**2 - 10"),
                        ("a2", "x**2 + (y - 3)**2 - 5")], 4.0),
        4.0, "anchors (0,0),(4,0),(0,3); measured ranges consistent at (1,1)"))

    # 4. Rosenbrock stationary point: gradient of the classic banana function.
    S.append(RealSystem(
        "rosenbrock_grad", "optimization",
        _g(["x", "y"], [("gx", "-2*(1 - x) - 400*x*(y - x**2)"),
                        ("gy", "200*(y - x**2)")], 2.0),
        2.0, "grad of (1-x)^2+100(y-x^2)^2 = 0; unique real root (1,1)"))

    # 5. Himmelblau stationary points: gradient of the four-minimum test function.
    S.append(RealSystem(
        "himmelblau_grad", "optimization",
        _g(["x", "y"], [("gx", "4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)"),
                        ("gy", "2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)")], 5.0),
        5.0, "grad of Himmelblau = 0; nine real critical points incl. (3,2)"))

    # 6. Two-link (2R) planar inverse kinematics: reach target (1,1) with unit links.
    #    Unknowns are (c1,s1,c2,s2) = cosines/sines of the two joint angles.
    S.append(RealSystem(
        "inverse_kinematics_2r", "robotics",
        _g(["c1", "s1", "c2", "s2"],
           [("u1", "c1**2 + s1**2 - 1"),
            ("u2", "c2**2 + s2**2 - 1"),
            ("px", "c1 + (c1*c2 - s1*s2) - 1"),
            ("py", "s1 + (s1*c2 + c1*s2) - 1")], 1.0),
        1.0, "2R arm, unit links, target (1,1) within reach (dist sqrt2 < 2)"))

    # 7. Three-link (3R) planar inverse kinematics: position + orientation, 6 unknowns,
    #    more strongly coupled and higher-dimensional than 2R.
    S.append(RealSystem(
        "inverse_kinematics_3r", "robotics",
        _g(["c1", "s1", "c2", "s2", "c3", "s3"],
           [("u1", "c1**2 + s1**2 - 1"),
            ("u2", "c2**2 + s2**2 - 1"),
            ("u3", "c3**2 + s3**2 - 1"),
            # x = c1 + cos(12) + cos(123); using product-to-sum expansions:
            ("px", "c1 + (c1*c2 - s1*s2) + "
                   "((c1*c2 - s1*s2)*c3 - (s1*c2 + c1*s2)*s3) - 2"),
            ("py", "s1 + (s1*c2 + c1*s2) + "
                   "((s1*c2 + c1*s2)*c3 + (c1*c2 - s1*s2)*s3) - 1"),
            # orientation theta1+theta2+theta3 = 0  =>  sin = 0 branch:
            ("po", "(s1*c2 + c1*s2)*c3 + (c1*c2 - s1*s2)*s3")], 1.0),
        1.0, "3R arm, unit links, target position (2,1), end orientation 0"))

    # 8. Cyclic-4: the standard computer-algebra benchmark (Groebner/homotopy suites).
    S.append(RealSystem(
        "cyclic4", "algebra_benchmark",
        _g(["a", "b", "c", "d"],
           [("e1", "a + b + c + d"),
            ("e2", "a*b + b*c + c*d + d*a"),
            ("e3", "a*b*c + b*c*d + c*d*a + d*a*b"),
            ("e4", "a*b*c*d - 1")], 1.5),
        1.5, "cyclic-4 roots system; has real solutions (standard benchmark)"))

    return S
