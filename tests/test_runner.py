"""Tests for marc/eval/runner.py — real Checker-based accept/reject and pass@k."""

import json

from marc.eval.runner import Problem, run_eval, run_split_eval, perturb_constants
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


class _InfoSolver(_ScriptedSolver):
    """Scripted solver that also exposes the optional sample_with_info hook."""

    def sample_with_info(self, problem, k):
        samples = self.sample(problem, k)
        infos = [
            {
                "n_steps": 7,
                "best_energy": 0.0,
                "final_energy": 0.0,
                "converged": True,
                "energies": [4.0, 1.0, 0.0],
            }
            for _ in samples
        ]
        return samples, infos


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


def test_runner_records_solve_info_when_solver_provides_it():
    # first candidate wrong, second is the solution -> first_success_index == 1
    solver = _InfoSolver([[2.5, 1.0], [2.0, 1.0]])
    metrics = run_eval([_problem("a")], solver=solver, n_samples=2)
    pp = metrics["per_problem"][0]
    assert pp["first_success_index"] == 1
    assert pp["wall_ms"] >= 0.0
    assert pp["n_steps"] == 7
    assert pp["final_energy"] == 0.0
    assert pp["energies"] == [4.0, 1.0, 0.0]
    # summary-level additions
    assert metrics["wall_ms_total"] >= 0.0
    assert metrics["wall_ms_mean"] >= 0.0
    curve = metrics["restart_curve"]
    assert [row["k"] for row in curve] == [1, 2]
    assert [row["rate"] for row in curve] == [0.0, 1.0]
    json.dumps(metrics)


def test_first_success_index_none_when_never_accepted():
    metrics = run_eval([_problem("a")], solver=_InfoSolver([[9.0, 9.0]]))
    assert metrics["per_problem"][0]["first_success_index"] is None


def test_runner_without_info_hook_keeps_schema_with_null_info():
    solver = _ScriptedSolver([[2.0, 1.0]])
    metrics = run_eval([_problem("a")], solver=solver)
    pp = metrics["per_problem"][0]
    # old schema keys unchanged (superset check)
    assert {"id", "accepted", "candidate", "max_residual", "reject_stage"} <= set(pp)
    assert pp["accepted"] is True
    # new keys present but null where the solver had nothing to report
    assert pp["n_steps"] is None
    assert pp["final_energy"] is None
    assert pp["energies"] is None
    assert pp["first_success_index"] == 0
    assert pp["wall_ms"] >= 0.0
    assert "restart_curve" in metrics and "wall_ms_total" in metrics


def test_split_eval_gains_per_split_instrumentation():
    solver = _InfoSolver([[2.0, 1.0]])
    metrics = run_split_eval([_problem("id1")], [_problem("ho1")], solver=solver)
    for split in metrics["splits"].values():
        assert split["wall_ms_total"] >= 0.0
        assert split["wall_ms_mean"] >= 0.0
        assert split["restart_curve"][0]["rate"] == 1.0
    # per-problem records across splits carry the info keys too
    assert all("wall_ms" in pp and "energies" in pp for pp in metrics["per_problem"])
    json.dumps(metrics)
