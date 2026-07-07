import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package, so load the module directly from its file path.
_SPEC = importlib.util.spec_from_file_location(
    "demo_end_to_end", Path(__file__).resolve().parent.parent / "scripts" / "demo_end_to_end.py"
)
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)


def _run(monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", ["demo_end_to_end.py"] + argv)
    demo.main()
    return capsys.readouterr().out


def test_default_falls_back_to_geometry_and_solves(monkeypatch, capsys):
    out = _run(monkeypatch, [], capsys)
    assert "built-in geometry example" in out
    assert "ACCEPTED" in out


def test_explicit_algebra_text_solves(monkeypatch, capsys):
    out = _run(monkeypatch, ["--text", "x plus y equals 5 and x minus y equals 1"], capsys)
    assert "x=3, y=2" in out
    assert "ACCEPTED" in out


def test_unparseable_text_falls_back_and_still_solves(monkeypatch, capsys):
    out = _run(monkeypatch, ["--text", "what is the meaning of life"], capsys)
    assert "could not parse" in out
    assert "ACCEPTED" in out


def test_learned_solver_out_of_distribution_reports_no_candidate_instead_of_crashing(monkeypatch, capsys):
    """Regression test for the None-candidate path (see marc/eval/solver.py's
    LearnedSolver.sample() docstring) — an untrained/OOD checkpoint should produce a
    clean message, not an unhandled exception."""
    pytest.importorskip("torch_geometric")

    out = _run(monkeypatch, ["--solver", "learned"], capsys)
    # Either it reports the clean "no candidate" message, or it reports a (likely
    # rejected) candidate — either way it must not raise, and must say something.
    assert out.strip() != ""
