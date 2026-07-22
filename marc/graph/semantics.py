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


def poly_summary(
    expression: str | sp.Poly, symbols: list[sp.Symbol] | None = None
) -> tuple[float, float, float, float, float]:
    """Shared polynomial summary: ``(degree, terms, has_cross, has_square, constant)``.

    Used by both this module's per-factor features and the repair ranker's
    candidate features (``marc.model.repair_ranker.candidate_features``) so the
    two views of "what kind of polynomial is this" cannot silently drift apart.
    ``degree`` is total_degree/4, ``terms`` is log1p(term count), ``has_cross``/
    ``has_square`` are 0/1 flags, and ``constant`` is the signed-log constant term.

    ``expression`` may be a pre-built ``sp.Poly`` (reused as-is, no re-parsing) or
    a raw expression string, in which case ``symbols`` are the generators to build
    the polynomial over (defaulting to the expression's own free symbols, sorted
    by name). Falls back to an all-zero summary — rather than raising — if
    ``expression`` doesn't parse as a polynomial, e.g. a non-polynomial defining
    expression.
    """
    if isinstance(expression, sp.Poly):
        poly = expression
    else:
        if symbols is None:
            try:
                symbols = sorted(sp.sympify(expression).free_symbols, key=lambda s: s.name)
            except (sp.SympifyError, TypeError, ValueError):
                symbols = []
        poly = _poly(expression, symbols)
    if poly is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    terms = poly.terms()
    degree = float(max(int(poly.total_degree()), 0)) / 4.0
    term_count = math.log1p(len(terms))
    has_cross = float(any(sum(e > 0 for e in m) >= 2 for m, _ in terms))
    has_square = float(any(any(e >= 2 for e in m) for m, _ in terms))
    constant = _slog(float(poly.eval({s: 0 for s in poly.gens})))
    return (degree, term_count, has_cross, has_square, constant)


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
        degree, term_count, has_cross, has_square, slog_constant = poly_summary(poly)
        arity = sum(any(m[i] for m, _ in terms) for i in range(len(symbols)))
        coeff_l1 = sum(abs(float(c)) for _, c in terms)
        factor_features.append([
            slog_constant,
            math.log1p(abs(constant)),
            degree,
            term_count,
            arity / 8.0,
            has_cross,
            has_square,
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
