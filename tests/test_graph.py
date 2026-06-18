from marc.graph.serialize import load_graph
from marc.graph.pyg import build_heterodata


def test_json_load():
    graph = load_graph(
        "marc/data/examples/two_equations.json"
    )

    assert len(graph.variables) == 2
    assert len(graph.factors) == 2
    assert len(graph.edges) == 4


def test_values():
    graph = load_graph(
        "marc/data/examples/two_equations.json"
    )

    graph.set_values([5, 7])

    assert graph.get_values().tolist() == [5.0, 7.0]


def test_pyg_build():
    graph = load_graph(
        "marc/data/examples/two_equations.json"
    )

    data = build_heterodata(graph)

    assert "variable" in data.node_types
    assert "factor" in data.node_types

    assert (
        data[
            "variable",
            "connected_to",
            "factor"
        ].edge_index.shape[1]
        == 4
    )