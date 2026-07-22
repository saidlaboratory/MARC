"""Operator-aware polynomial features for learned constraint proposals.

The original PyG conversion exposes only a factor's constant term and one
hand-authored scalar per incidence edge.  Consequently, ``x + y - 3`` and
``x*y - 3`` can be identical neural inputs.  This module keeps the existing
conversion untouched for checkpoint compatibility and supplies a richer view
for new proposal models.

The representation is deliberately small and deterministic: factor features
summarize polynomial degree/terms/operator structure, while edge features
describe how each incident variable participates (linear, square, and cross
terms).  All unbounded numeric features use signed ``log1p`` scaling.
"""

from __future__ import annotations

import math

import sympy as sp
import torch
from torch_geometric.data import HeteroData

from .graph import FactorGraph


FACTOR_FEATURE_DIM = 8
EDGE_FEATURE_DIM = 6


def _slog(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _poly(expression: str, symbols: list[sp.Symbol]):
    try:
        return sp.Poly(sp.sympify(expression), *symbols)
    except (sp.PolynomialError, sp.SympifyError, TypeError, ValueError):
        return None


def build_semantic_heterodata(graph: FactorGraph) -> HeteroData:
    """Convert ``graph`` to operator-aware heterogeneous tensors.

    Shapes:
      - ``variable.x``: ``[V,1]`` current/noised values
      - ``factor.x``: ``[F,8]`` polynomial summaries
      - incidence ``edge_attr``: ``[E,6]`` variable-in-factor summaries
    """
    data = HeteroData()
    variables = list(graph.variables)
    factors = list(graph.factors)
    symbols = [sp.Symbol(v.id) for v in variables]
    symbol_index = {v.id: i for i, v in enumerate(variables)}
    factor_index = {f.id: i for i, f in enumerate(factors)}

    data["variable"].x = torch.tensor(
        [[float(v.value)] for v in variables], dtype=torch.float32
    )

    polys = [_poly(f.expression, symbols) for f in factors]
    factor_features = []
    for poly in polys:
        if poly is None:
            factor_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            continue
        terms = poly.terms()
        constant = float(poly.eval({s: 0 for s in symbols}))
        degree = max(int(poly.total_degree()), 0)
        arity = sum(any(m[i] for m, _ in terms) for i in range(len(symbols)))
        has_square = any(any(e >= 2 for e in monom) for monom, _ in terms)
        has_cross = any(sum(e > 0 for e in monom) >= 2 for monom, _ in terms)
        coeff_l1 = sum(abs(float(c)) for _, c in terms)
        factor_features.append([
            _slog(constant),
            math.log1p(abs(constant)),
            degree / 4.0,
            math.log1p(len(terms)),
            arity / 8.0,
            float(has_cross),
            float(has_square),
            math.log1p(coeff_l1),
        ])
    data["factor"].x = torch.tensor(factor_features, dtype=torch.float32).reshape(
        len(factors), FACTOR_FEATURE_DIM
    )

    src, dst, edge_features = [], [], []
    for edge in graph.edges:
        vi = symbol_index[edge.variable_id]
        fi = factor_index[edge.factor_id]
        poly = polys[fi]
        linear = diagonal_quadratic = participation = 0.0
        max_exponent = cross = 0.0
        if poly is not None:
            for monom, coeff in poly.terms():
                exponent = int(monom[vi])
                if exponent == 0:
                    continue
                c = float(coeff)
                total_degree = sum(monom)
                participation += c
                max_exponent = max(max_exponent, float(exponent))
                if total_degree == 1 and exponent == 1:
                    linear += c
                if total_degree == 2 and exponent == 2:
                    diagonal_quadratic += c
                if sum(e > 0 for e in monom) >= 2:
                    cross += abs(c)
        src.append(vi)
        dst.append(fi)
        edge_features.append([
            _slog(linear),
            _slog(diagonal_quadratic),
            max_exponent / 4.0,
            _slog(participation),
            math.log1p(cross),
            _slog(float(getattr(edge, "coefficient", 1.0))),
        ])

    store = data["variable", "connected_to", "factor"]
    store.edge_index = torch.tensor([src, dst], dtype=torch.long)
    store.edge_attr = torch.tensor(edge_features, dtype=torch.float32).reshape(
        len(src), EDGE_FEATURE_DIM
    )
    return data
