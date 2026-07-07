from marc.cas.checker import Checker
from marc.data.geometry import build_linked_graph, build_triangle_graph
from marc.eval.problems import geometry_held_out, geometry_in_distribution


def test_geometry_in_distribution_accepts_its_own_solution():
    checker = Checker()
    for p in geometry_in_distribution(n=20):
        assert p.metadata["split"] == "geometry_in_distribution"
        assert len(p.graph.variables) == 2
        assert checker.accepts(p.graph, p.solution), (p.description, p.solution)


def test_geometry_held_out_accepts_its_own_solution():
    checker = Checker()
    for p in geometry_held_out(n=20):
        assert p.metadata["split"] == "geometry_held_out"
        assert len(p.graph.variables) == 4
        assert checker.accepts(p.graph, p.solution), (p.description, p.solution)


def test_build_triangle_graph_shape():
    graph = build_triangle_graph(b_sq=18, a_sq=10, c=4)
    assert [v.id for v in graph.variables] == ["x", "y"]
    assert len(graph.factors) == 2


def test_build_linked_graph_shape():
    graph = build_linked_graph(b1_sq=18, a1_sq=10, link_sq=5, a2_sq=13, c=4)
    assert [v.id for v in graph.variables] == ["x1", "y1", "x2", "y2"]
    assert len(graph.factors) == 4
