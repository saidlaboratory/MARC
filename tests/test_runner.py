"""Tests for marc/eval/runner.py — real Checker-based accept/reject and pass@k."""

import json

from marc.eval.runner import Problem, run_eval, perturb_constants
from marc.graph.graph import FactorGraph
from marc.graph.schema import VariableNode, FactorNode


def _problem(pid: str) -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[FactorNode("eq1", "x+y-3"), FactorNode("eq2", "x-y-1")],
        edges=[],
    )
    return Problem(id=pid, graph=graph, solution=[2.0, 1.0])


class _ScriptedSolver:
    """Returns a fixed list of candidates per call, regardless of problem."""

    def __init__(self, candidates):
        self._candidates = candidates

    def sample(self, problem, k):
        return [list(c) for c in self._candidates[:k]]


def test_runner_reports_real_acceptance():
    problems = [_problem("a"), _problem("b")]
    solver = _ScriptedSolver([[2.0, 1.0]])  # exact solution
    metrics = run_eval(problems, solver=solver)
    assert metrics["solve_rate"] == 1.0
    assert all(pp["accepted"] for pp in metrics["per_problem"])


def test_runner_rejects_wrong_answer_with_real_residual():
    metrics = run_eval([_problem("a")], solver=_ScriptedSolver([[2.5, 1.0]]))
    assert metrics["solve_rate"] == 0.0
    pp = metrics["per_problem"][0]
    assert pp["reject_stage"] == "numeric"
    assert pp["max_residual"] > 0


def test_pass_at_k_beats_pass_at_1():
    # first sample wrong, second is the solution -> pass@1 = 0, pass@2 = 1.
    solver = _ScriptedSolver([[2.5, 1.0], [2.0, 1.0]])
    metrics = run_eval([_problem("a")], solver=solver, n_samples=2)
    assert metrics["solve_rate"] == 0.0      # first candidate failed
    assert metrics["pass_at_k"] == 1.0       # a later candidate solved it


def test_metrics_are_json_serialisable():
    metrics = run_eval([_problem("a")], solver=_ScriptedSolver([[2.0, 1.0]]))
    json.dumps(metrics)  # must not raise
