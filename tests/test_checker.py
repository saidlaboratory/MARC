"""Tests for marc/cas/checker.py — the two-stage accept/reject gate."""

import pytest
import sympy as sp

from marc.cas.checker import Checker, CheckResult
from marc.cas.engine import CASEngine
from marc.graph.graph import FactorGraph
from marc.graph.serialize import load_graph
from marc.graph.schema import VariableNode, FactorNode, Edge


def two_equations_graph() -> FactorGraph:
    """x + y - 3 = 0 and x - y - 1 = 0, unique solution (x=2, y=1)."""
    return FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("eq1", "x+y-3"), FactorNode("eq2", "x-y-1")],
        edges=[
            Edge("x", "eq1", 1), Edge("y", "eq1", 1),
            Edge("x", "eq2", 1), Edge("y", "eq2", -1),
        ],
    )


def test_accepts_exact_solution():
    result = Checker().verify(two_equations_graph(), [2.0, 1.0])
    assert isinstance(result, CheckResult)
    assert result.accepted
    assert result.failed_factors == []
    assert result.max_residual == pytest.approx(0.0, abs=1e-12)


def test_rejects_wrong_solution_numeric_stage():
    # (2.5, 1.0): residuals 0.5 — fails the numeric pre-filter outright.
    result = Checker().verify(two_equations_graph(), [2.5, 1.0])
    assert not result.accepted
    assert result.stage == "numeric"
    assert set(result.failed_factors) == {"eq1", "eq2"}
    assert result.max_residual == pytest.approx(0.5)


def test_float_false_accept_caught_by_symbolic_stage():
    # (2.0000005, 1.0): residual 5e-7 slips past the numeric gate at tol=1e-6,
    # but exact arithmetic shows residual = 1/2000000 != 0 -> symbolic rejects.
    x = [2.0000005, 1.0]
    checker = Checker(tol=1e-6)

    # A naive numeric-only gate would have accepted this: both violations < tol.
    symbols = [sp.Symbol("x"), sp.Symbol("y")]
    factors = [("eq1", sp.sympify("x+y-3")), ("eq2", sp.sympify("x-y-1"))]
    violations = checker._numeric_violations(symbols, factors, x)
    assert all(v <= checker.tol for _, v in violations)  # numeric gate alone is fooled

    result = checker.verify(two_equations_graph(), x)
    assert not result.accepted  # but the two-stage checker is not
    assert result.stage == "symbolic"
    assert set(result.failed_factors) == {"eq1", "eq2"}


def test_robust_to_genuine_float_noise():
    # Values within snap_tol of the true solution snap back and are accepted.
    result = Checker().verify(two_equations_graph(), [1.9999999999, 1.0000000001])
    assert result.accepted


def test_non_finite_residual_is_rejected():
    # A diverged sample (NaN/inf) must be rejected, not slip through abs(nan) > tol.
    for bad in (float("nan"), float("inf")):
        result = Checker().verify(two_equations_graph(), [bad, 1.0])
        assert not result.accepted
        assert result.stage == "numeric"


def test_empty_graph_is_vacuously_accepted():
    g = FactorGraph(variables=[VariableNode("x")], factors=[], edges=[])
    result = Checker().verify(g, [1.0])
    assert result.accepted
    assert result.max_residual == 0.0


def test_complex_residual_is_rejected_not_crash():
    # sqrt(x) - 2 at x=-1 is non-real; must reject cleanly, not raise.
    g = FactorGraph(
        variables=[VariableNode("x")],
        factors=[FactorNode("r", "sqrt(x) - 2")],
        edges=[],
    )
    result = Checker().verify(g, [-1.0])
    assert not result.accepted
    assert result.stage == "numeric"


def test_radical_residual_not_false_rejected():
    # sqrt(x) - 2*sqrt(2) is exactly 0 at x=8; the simplify fallback must accept it.
    g = FactorGraph(
        variables=[VariableNode("x")],
        factors=[FactorNode("r", "sqrt(x) - 2*sqrt(2)")],
        edges=[],
    )
    assert Checker().accepts(g, [8.0])
    assert not Checker().accepts(g, [9.0])


def test_inequality_constraint():
    # x <= 5  and  x >= 1, both as relational factor expressions.
    g = FactorGraph(
        variables=[VariableNode("x")],
        factors=[FactorNode("ub", "x - 5 <= 0"), FactorNode("lb", "x >= 1")],
        edges=[],
    )
    checker = Checker()
    assert checker.accepts(g, [3.0])          # inside the band
    assert checker.accepts(g, [5.0])          # on the (non-strict) boundary
    result = checker.verify(g, [7.0])         # above upper bound
    assert not result.accepted
    assert result.failed_factors == ["ub"]


def test_mixed_equality_and_inequality():
    # x + y = 3 (equality) with x >= 0 (inequality); solution (2, 1) satisfies both.
    g = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("sum", "x + y - 3"), FactorNode("pos", "x >= 0")],
        edges=[],
    )
    assert Checker().accepts(g, [2.0, 1.0])
    assert not Checker().accepts(g, [-1.0, 4.0])  # equality ok, but x >= 0 violated


def test_first_accepted_picks_first_passing_candidate():
    g = two_equations_graph()
    checker = Checker()
    # third candidate is the solution; first two are wrong.
    candidates = [[0.0, 0.0], [2.5, 1.0], [2.0, 1.0], [2.0, 1.0]]
    hit = checker.first_accepted(g, candidates)
    assert hit is not None
    index, result = hit
    assert index == 2
    assert result.accepted


def test_first_accepted_returns_none_when_all_fail():
    g = two_equations_graph()
    assert Checker().first_accepted(g, [[0.0, 0.0], [2.5, 1.0]]) is None


def test_unknown_variable_raises_clear_error():
    g = FactorGraph(
        variables=[VariableNode("x")],
        factors=[FactorNode("bad", "x + z - 1")],  # z is undeclared
        edges=[],
    )
    with pytest.raises(ValueError, match="unknown variables"):
        Checker().verify(g, [1.0])


def test_explain_rejection_is_readable():
    text = Checker().explain_rejection(two_equations_graph(), [2.5, 1.0])
    assert "REJECTED" in text
    assert "eq1" in text and "eq2" in text
    assert "x=2.5" in text


def test_cas_engine_accepts_delegates_to_checker():
    engine = CASEngine("marc/data/examples/two_equations.json", "x y")
    assert engine.accepts([2.0, 1.0]) is True
    # false-accept that an energy<tol gate would wave through, now caught:
    assert engine.accepts([2.0000005, 1.0]) is False
    assert engine.accepts([2.5, 1.0]) is False


def test_cas_engine_verify_returns_full_result():
    engine = CASEngine("marc/data/examples/two_equations.json", "x y")
    result = engine.verify([2.5, 1.0])
    assert isinstance(result, CheckResult)
    assert not result.accepted
    assert result.stage == "numeric"
    assert result.max_residual == pytest.approx(0.5)


def test_loaded_graph_from_json_round_trips_through_checker():
    g = load_graph("marc/data/examples/two_equations.json")
    assert Checker().accepts(g, [2.0, 1.0])
    assert not Checker().accepts(g, [2.5, 1.0])


def test_check_result_to_dict_is_json_friendly():
    import json
    result = Checker().verify(two_equations_graph(), [2.0, 1.0])
    d = result.to_dict()
    assert d == {"accepted": True, "failed_factors": [], "max_residual": 0.0, "stage": ""}
    json.dumps(d)  # must not raise
