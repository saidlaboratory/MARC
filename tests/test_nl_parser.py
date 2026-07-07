import pytest

from marc.cas.checker import Checker
from marc.nl.parser import NLParseError, parse


def test_sum_diff_worded():
    graph, sol = parse("x plus y equals 5 and x minus y equals 1")
    assert [v.id for v in graph.variables] == ["x", "y"]
    assert sol == {"x": 3.0, "y": 2.0}
    assert Checker().accepts(graph, [sol["x"], sol["y"]])


def test_sum_diff_symbolic():
    graph, sol = parse("x + y = 5 and x - y = 1")
    assert sol == {"x": 3.0, "y": 2.0}
    assert Checker().accepts(graph, [3.0, 2.0])


def test_sum_product_has_no_closed_form_solution():
    graph, sol = parse("x plus y equals -1 and x times y equals -12")
    assert sol == {}
    assert Checker().accepts(graph, [3.0, -4.0])  # 3 + -4 = -1, 3 * -4 = -12


def test_geometry_sentence():
    graph, sol = parse(
        "A point is at squared distance 18 from the origin and squared distance 10 from (4, 0)."
    )
    assert [v.id for v in graph.variables] == ["x", "y"]
    assert sol == {}
    assert Checker().accepts(graph, [3.0, 3.0])


def test_unparseable_text_raises():
    with pytest.raises(NLParseError):
        parse("what is the meaning of life")
