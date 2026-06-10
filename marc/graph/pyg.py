import torch
from torch_geometric.data import HeteroData

from .graph import FactorGraph


def build_heterodata(graph: FactorGraph):
    data = HeteroData()

    data["variable"].x = torch.tensor(
        [[v.value] for v in graph.variables],
        dtype=torch.float,
    )

    data["factor"].x = torch.zeros(
        (len(graph.factors), 1),
        dtype=torch.float,
    )

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

    for edge in graph.edges:
        src.append(variable_map[edge.variable_id])
        dst.append(factor_map[edge.factor_id])

    data[
        "variable",
        "connected_to",
        "factor"
    ].edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long,
    )

    return data