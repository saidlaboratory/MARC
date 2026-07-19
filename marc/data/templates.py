"""Problem templates for procedural math problem generation."""

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

_NONZERO = [-3, -2, -1, 1, 2, 3]


def _build_expr(coeffs: list, var_names: list, rhs: int) -> str:
    """Build a SymPy-compatible expression string that equals zero.

    For example, coeffs=[2, -3], var_names=["x", "y"], rhs=5
    produces "2*x-3*y-5".
    """
    parts = []
    for coef, name in zip(coeffs, var_names):
        if coef == 1:
            parts.append(f"+{name}")
        elif coef == -1:
            parts.append(f"-{name}")
        elif coef > 0:
            parts.append(f"+{coef}*{name}")
        else:
            parts.append(f"{coef}*{name}")
    if rhs > 0:
        parts.append(f"-{rhs}")
    elif rhs < 0:
        parts.append(f"+{-rhs}")
    return "".join(parts).lstrip("+")


@dataclass
class LinearSystem2x2Template:
    """2-variable linear system: a1*x + b1*y = c1, a2*x + b2*y = c2."""

    name: str = "LinearSystem2x2"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        """Generate a random 2x2 linear system with a known integer solution.

        Algorithm:
        1. Pick x*, y* as random integers in [-5, 5].
        2. Pick random non-zero coefficients a1, b1, a2, b2 in [-3,3] \\ {0}.
        3. Compute c1 = a1*x* + b1*y*, c2 = a2*x* + b2*y*.
        4. Reject if the system is degenerate (det == 0), and retry.
        5. Build FactorGraph with expressions like "a1*x+b1*y-c1".

        Returns:
            (FactorGraph with initial values 0.0, {"x": x*, "y": y*})
        """
        rng = random.Random(seed)

        for _ in range(100):
            x_star = rng.randint(-5, 5)
            y_star = rng.randint(-5, 5)
            a1 = rng.choice(_NONZERO)
            b1 = rng.choice(_NONZERO)
            a2 = rng.choice(_NONZERO)
            b2 = rng.choice(_NONZERO)

            if a1 * b2 - a2 * b1 == 0:
                continue  # degenerate — retry

            c1 = a1 * x_star + b1 * y_star
            c2 = a2 * x_star + b2 * y_star

            variables = [
                VariableNode(id="x", value=0.0),
                VariableNode(id="y", value=0.0),
            ]
            factors = [
                FactorNode(id="eq1", expression=_build_expr([a1, b1], ["x", "y"], c1)),
                FactorNode(id="eq2", expression=_build_expr([a2, b2], ["x", "y"], c2)),
            ]
            edges = [
                Edge("x", "eq1", float(a1)),
                Edge("y", "eq1", float(b1)),
                Edge("x", "eq2", float(a2)),
                Edge("y", "eq2", float(b2)),
            ]
            graph = FactorGraph(variables=variables, factors=factors, edges=edges)
            solution = {"x": float(x_star), "y": float(y_star)}
            return graph, solution

        raise RuntimeError("Could not generate a non-degenerate 2x2 linear system")


@dataclass
class LinearSystem3x3Template:
    """3-variable linear system: a*x + b*y + c*z = d for 3 equations."""

    name: str = "LinearSystem3x3"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        """Generate a random 3x3 linear system with a known integer solution.

        Similar to 2x2 but with 3 variables x, y, z and 3 equations.
        Uses numpy to check det != 0. Retries if degenerate.

        Returns:
            (FactorGraph with initial values 0.0, {"x": x*, "y": y*, "z": z*})
        """
        rng = random.Random(seed)
        var_names = ["x", "y", "z"]

        for _ in range(100):
            x_star = rng.randint(-5, 5)
            y_star = rng.randint(-5, 5)
            z_star = rng.randint(-5, 5)

            A = [[rng.choice(_NONZERO) for _ in range(3)] for _ in range(3)]
            if abs(np.linalg.det(A)) < 1e-9:
                continue  # degenerate — retry

            star = [x_star, y_star, z_star]
            rhs = [sum(A[i][j] * star[j] for j in range(3)) for i in range(3)]

            variables = [VariableNode(id=name, value=0.0) for name in var_names]
            factors = [
                FactorNode(
                    id=f"eq{i + 1}",
                    expression=_build_expr(A[i], var_names, rhs[i]),
                )
                for i in range(3)
            ]
            edges = [
                Edge(var_names[j], f"eq{i + 1}", float(A[i][j]))
                for i in range(3)
                for j in range(3)
            ]
            graph = FactorGraph(variables=variables, factors=factors, edges=edges)
            solution = {"x": float(x_star), "y": float(y_star), "z": float(z_star)}
            return graph, solution

        raise RuntimeError("Could not generate a non-degenerate 3x3 linear system")


# ---------------------------------------------------------------------------
# Coordinate geometry (P4, Davin) — points as variables, distance/slope factors
# ---------------------------------------------------------------------------
#
# Extends the generator beyond linear equation systems to 2-D coordinate geometry.
# A point becomes a pair of variable nodes (px, py); relations become factor
# expressions:
#   - pin      : px - a                          (anchor a known coordinate)
#   - distance : (px-qx)**2 + (py-qy)**2 - d2    (squared distance == d2)
#   - slope    : rise*(px-ax) - run*(py-ay)      (P on the line through A with
#                                                  direction (run, rise))
# Coordinates are integers, so every residual is an exact integer and the CAS /
# checker accept the stored solution exactly (energy == 0).


def _pin_expr(var_id: str, value: int) -> str:
    """Factor 'var - value' pinning a coordinate to a known integer."""
    return f"{var_id} - ({value})"


def _distance_expr(x1: str, y1: str, x2: str, y2: str, d2: int) -> str:
    """Squared-distance equality: (x1-x2)**2 + (y1-y2)**2 - d2 == 0."""
    return f"({x1}-{x2})**2 + ({y1}-{y2})**2 - ({d2})"


def _slope_expr(px: str, py: str, ax: str, ay: str, rise: int, run: int) -> str:
    """Collinearity of P with the line through A of direction (run, rise):
    rise*(px-ax) - run*(py-ay) == 0."""
    return f"({rise})*({px}-{ax}) - ({run})*({py}-{ay})"


def _point_edges(factor_id: str, var_ids) -> list:
    """One unit-coefficient edge per variable appearing in a (nonlinear) factor."""
    return [Edge(v, factor_id, 1.0) for v in var_ids]


@dataclass
class TriangleDistanceTemplate:
    """Coordinate geometry: find vertex C given base A, B and its distances to each.

    Points A, B, C are variables; A and B are pinned to known integer coordinates and
    two squared-distance factors fix C. A well-posed 6-variable / 6-factor system whose
    stored integer solution the CAS accepts exactly.
    """

    name: str = "TriangleDistance"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        rng = random.Random(seed)
        for _ in range(200):
            ax, ay = rng.randint(-5, 5), rng.randint(-5, 5)
            bx, by = rng.randint(-5, 5), rng.randint(-5, 5)
            cx, cy = rng.randint(-5, 5), rng.randint(-5, 5)
            if (ax, ay) == (bx, by) or (cx, cy) in {(ax, ay), (bx, by)}:
                continue  # points must be distinct
            # C off line AB (non-degenerate triangle): twice the signed area != 0
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) == 0:
                continue

            d_ac2 = (cx - ax) ** 2 + (cy - ay) ** 2
            d_bc2 = (cx - bx) ** 2 + (cy - by) ** 2

            var_ids = ["ax", "ay", "bx", "by", "cx", "cy"]
            variables = [VariableNode(id=v, value=0.0) for v in var_ids]
            factors = [
                FactorNode("pin_ax", _pin_expr("ax", ax)),
                FactorNode("pin_ay", _pin_expr("ay", ay)),
                FactorNode("pin_bx", _pin_expr("bx", bx)),
                FactorNode("pin_by", _pin_expr("by", by)),
                FactorNode("dist_ac", _distance_expr("cx", "cy", "ax", "ay", d_ac2)),
                FactorNode("dist_bc", _distance_expr("cx", "cy", "bx", "by", d_bc2)),
            ]
            edges = (
                _point_edges("pin_ax", ["ax"])
                + _point_edges("pin_ay", ["ay"])
                + _point_edges("pin_bx", ["bx"])
                + _point_edges("pin_by", ["by"])
                + _point_edges("dist_ac", ["cx", "cy", "ax", "ay"])
                + _point_edges("dist_bc", ["cx", "cy", "bx", "by"])
            )
            graph = FactorGraph(variables=variables, factors=factors, edges=edges)
            solution = {
                "ax": float(ax), "ay": float(ay),
                "bx": float(bx), "by": float(by),
                "cx": float(cx), "cy": float(cy),
            }
            return graph, solution

        raise RuntimeError("Could not generate a non-degenerate triangle")


@dataclass
class PointSlopeTemplate:
    """Coordinate geometry: find P on the line through anchor A at a given distance.

    A is pinned; P is fixed by one slope (collinearity) factor and one squared-distance
    factor — a 4-variable / 4-factor system exercising a slope factor alongside distance.
    """

    name: str = "PointSlope"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        rng = random.Random(seed)
        for _ in range(200):
            ax, ay = rng.randint(-5, 5), rng.randint(-5, 5)
            px, py = rng.randint(-5, 5), rng.randint(-5, 5)
            if (px, py) == (ax, ay):
                continue  # P must differ from A (nonzero distance & defined direction)

            rise = py - ay
            run = px - ax
            d_ap2 = run ** 2 + rise ** 2

            var_ids = ["ax", "ay", "px", "py"]
            variables = [VariableNode(id=v, value=0.0) for v in var_ids]
            factors = [
                FactorNode("pin_ax", _pin_expr("ax", ax)),
                FactorNode("pin_ay", _pin_expr("ay", ay)),
                FactorNode("slope_ap", _slope_expr("px", "py", "ax", "ay", rise, run)),
                FactorNode("dist_ap", _distance_expr("px", "py", "ax", "ay", d_ap2)),
            ]
            edges = (
                _point_edges("pin_ax", ["ax"])
                + _point_edges("pin_ay", ["ay"])
                + _point_edges("slope_ap", ["px", "py", "ax", "ay"])
                + _point_edges("dist_ap", ["px", "py", "ax", "ay"])
            )
            graph = FactorGraph(variables=variables, factors=factors, edges=edges)
            solution = {
                "ax": float(ax), "ay": float(ay),
                "px": float(px), "py": float(py),
            }
            return graph, solution

        raise RuntimeError("Could not generate a valid point-slope problem")


#: The coordinate-geometry template family (P4).
GEOMETRY_TEMPLATES = [TriangleDistanceTemplate(), PointSlopeTemplate()]


# ---------------------------------------------------------------------------
# Hard (non-convex) templates — A1 de-saturation tier.
# Convex linear systems let the classical `refine` solver hit ~100% every time,
# so the eval suite saturates and H1 has no signal. These bilinear/quadratic
# families create spurious energy minima that trap plain gradient descent, pulling
# solve rates off the ceiling into the separable 0.3–0.8 band. Encodings mirror the
# P3 structure toys (marc/eval/structure_eval.py).
# ---------------------------------------------------------------------------


@dataclass
class BilinearSystemTemplate:
    """2-variable bilinear (sum/product) system:  x + y = s,  x*y = p.

    Non-convex: E = (x+y-s)^2 + (x*y-p)^2 has spurious stationary points that trap
    gradient descent from a cold start. Solutions are the Vieta pair {x*, y*}."""

    name: str = "BilinearSystem"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        rng = random.Random(seed)
        x_star = rng.choice(_NONZERO)
        y_star = rng.choice(_NONZERO)
        while y_star == x_star:  # avoid the degenerate x=y instance
            y_star = rng.choice(_NONZERO)
        s, p = x_star + y_star, x_star * y_star
        variables = [VariableNode("x", value=0.0), VariableNode("y", value=0.0)]
        factors = [
            FactorNode("eq1", f"x+y-({s})"),
            FactorNode("eq2", f"x*y-({p})"),
        ]
        edges = [Edge("x", "eq1", 1.0), Edge("y", "eq1", 1.0),
                 Edge("x", "eq2", 1.0), Edge("y", "eq2", 1.0)]
        graph = FactorGraph(variables=variables, factors=factors, edges=edges)
        return graph, {"x": float(x_star), "y": float(y_star)}


@dataclass
class BilinearProductTemplate:
    """3-variable bilinear system:  x*y = A,  y*z = B,  x*z = C.

    Strongly non-convex and coupled; the product manifold has multiple sign-flipped
    branches (x,y,z) and (-x,-y,z)-type reflections, so cold-start descent frequently
    stalls at an inconsistent point."""

    name: str = "BilinearProduct"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        rng = random.Random(seed)
        x_star = rng.choice(_NONZERO)
        y_star = rng.choice(_NONZERO)
        z_star = rng.choice(_NONZERO)
        A, B, C = x_star * y_star, y_star * z_star, x_star * z_star
        variables = [VariableNode("x", value=0.0), VariableNode("y", value=0.0),
                     VariableNode("z", value=0.0)]
        factors = [
            FactorNode("eq1", f"x*y-({A})"),
            FactorNode("eq2", f"y*z-({B})"),
            FactorNode("eq3", f"x*z-({C})"),
        ]
        edges = [Edge("x", "eq1", 1.0), Edge("y", "eq1", 1.0),
                 Edge("y", "eq2", 1.0), Edge("z", "eq2", 1.0),
                 Edge("x", "eq3", 1.0), Edge("z", "eq3", 1.0)]
        graph = FactorGraph(variables=variables, factors=factors, edges=edges)
        return graph, {"x": float(x_star), "y": float(y_star), "z": float(z_star)}


#: The hard non-convex template family (A1 de-saturation tier).
HARD_TEMPLATES = [BilinearSystemTemplate(), BilinearProductTemplate()]
