import json
from dataclasses import asdict

from .schema import VariableNode, FactorNode, Edge
from .graph import FactorGraph


def save_graph(graph: FactorGraph, path: str):
    data = {
        "variables": [asdict(v) for v in graph.variables],
        "factors": [asdict(f) for f in graph.factors],
        "edges": [asdict(e) for e in graph.edges],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str) -> FactorGraph:
    with open(path, "r") as f:
        data = json.load(f)

    variables = [
        VariableNode(**v)
        for v in data["variables"]
    ]

    factors = [
        FactorNode(**f)
        for f in data["factors"]
    ]

    edges = [
        Edge(**e)
        for e in data["edges"]
    ]

    return FactorGraph(
        variables=variables,
        factors=factors,
        edges=edges,
    )