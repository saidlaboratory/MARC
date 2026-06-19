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
