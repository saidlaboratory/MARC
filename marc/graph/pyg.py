import sympy as sp
import torch
from torch_geometric.data import HeteroData

from .graph import FactorGraph


def _factor_constant(expression: str, symbols: dict) -> float:
    """Constant term of a factor expression = its value with all variables at 0
    (encodes the RHS: for ``x + y - 3`` this is ``-3``). Falls back to 0.0 if the
    expression can't be parsed as a constant offset."""
    try:
        expr = sp.sympify(expression, locals=symbols)
        return float(expr.subs({s: 0 for s in symbols.values()}))
    except (sp.SympifyError, TypeError, ValueError):
        return 0.0


def build_heterodata(graph: FactorGraph):
    data = HeteroData()

    data["variable"].x = torch.tensor(
        [[v.value] for v in graph.variables],
        dtype=torch.float,
    )

    # Factor node feature = the constraint's constant term (its RHS), so systems
    # that differ only in their constants are distinguishable to the model.
    _syms = {v.id: sp.Symbol(v.id) for v in graph.variables}
    data["factor"].x = torch.tensor(
        [[_factor_constant(f.expression, _syms)] for f in graph.factors],
        dtype=torch.float,
    ).reshape(len(graph.factors), 1)

    variable_map = {
        v.id: i
        for i, v in enumerate(graph.variables)
    }

    factor_map = {
        f.id: i
        for i, f in enumerate(graph.factors)
    }

    src = []
    dst = []
    coeffs = []

    for edge in graph.edges:
        src.append(variable_map[edge.variable_id])
        dst.append(factor_map[edge.factor_id])
        coeffs.append([float(getattr(edge, "coefficient", 1.0))])

    edge_type = data["variable", "connected_to", "factor"]
    edge_type.edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type.edge_attr = torch.tensor(coeffs, dtype=torch.float)

    return data