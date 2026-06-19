import sympy as sp
import pytest
from marc.cas.engine import CASEngine
from marc.eval.checker import Checker, CheckResult

EXAMPLE_JSON = "marc/data/examples/two_equations.json"


def make_checker():
    cas = CASEngine(EXAMPLE_JSON, "x y")
    x, y = sp.symbols("x y")
    # x+y-3=0, x-y-1=0
    exprs = [x + y - 3, x - y - 1]
    return Checker(cas, exprs)


def test_numeric_accepts_solution():
    checker = make_checker()
    result = checker.check_numeric([2.0, 1.0])
    assert result.accepted
    assert result.gate == "numeric"


def test_numeric_rejects_wrong():
    checker = make_checker()
    result = checker.check_numeric([1.0, 1.0])  # wrong: 1+1=2≠3
    assert not result.accepted


def test_symbolic_accepts_solution():
    checker = make_checker()
    result = checker.check_symbolic([2.0, 1.0])
    assert result.accepted
    assert result.gate == "symbolic"


def test_symbolic_rejects_near_miss():
    checker = make_checker()
    # Numerically close but symbolically wrong
    result = checker.check_symbolic([2.0001, 0.9999])
    assert not result.accepted


def test_full_check_accepts_solution():
    checker = make_checker()
    result = checker.check([2.0, 1.0])
    assert result.accepted


def test_full_check_rejects_wrong():
    checker = make_checker()
    result = checker.check([0.0, 0.0])
    assert not result.accepted


def test_check_result_is_dataclass():
    r = CheckResult(accepted=True, gate="numeric", explanation="test")
    assert r.accepted
    assert r.gate == "numeric"


def test_numeric_gate_tolerance():
    checker = make_checker()
    # Slightly off solution — passes wide tol but fails tight tol
    result_loose = checker.check_numeric([2.0001, 1.0001], tol=0.01)
    result_tight = checker.check_numeric([2.0001, 1.0001], tol=1e-6)
    assert result_loose.accepted
    assert not result_tight.accepted
