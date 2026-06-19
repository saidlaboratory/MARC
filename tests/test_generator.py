"""Tests for procedural problem templates and ProblemGenerator."""

import json
import os
import tempfile

import pytest

from marc.cas.engine import CASEngine
from marc.data.generator import ProblemGenerator
from marc.data.templates import LinearSystem2x2Template, LinearSystem3x3Template
from marc.graph.serialize import load_graph, save_graph


# ---------------------------------------------------------------------------
# LinearSystem2x2Template
# ---------------------------------------------------------------------------


def test_2x2_template_generates_valid():
    tmpl = LinearSystem2x2Template()
    graph, solution = tmpl.generate(seed=42)
    assert len(graph.variables) == 2
    assert len(graph.factors) == 2
    assert len(graph.edges) == 4
    assert "x" in solution and "y" in solution


def test_2x2_cas_accepts_solution():
    tmpl = LinearSystem2x2Template()
    graph, solution = tmpl.generate(seed=42)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_graph(graph, tmp)
        sym = " ".join(v.id for v in graph.variables)
        cas = CASEngine(tmp, sym)
        x_vals = [solution[v.id] for v in graph.variables]
        assert cas.accepts(x_vals, tol=1e-4), f"Energy={cas.energy(x_vals)}"
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# LinearSystem3x3Template
# ---------------------------------------------------------------------------


def test_3x3_template_generates_valid():
    tmpl = LinearSystem3x3Template()
    graph, solution = tmpl.generate(seed=123)
    assert len(graph.variables) == 3
    assert len(graph.factors) == 3
    assert "x" in solution and "y" in solution and "z" in solution


def test_3x3_cas_accepts_solution():
    tmpl = LinearSystem3x3Template()
    graph, solution = tmpl.generate(seed=123)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_graph(graph, tmp)
        sym = " ".join(v.id for v in graph.variables)
        cas = CASEngine(tmp, sym)
        x_vals = [solution[v.id] for v in graph.variables]
        assert cas.accepts(x_vals, tol=1e-4), f"Energy={cas.energy(x_vals)}"
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# ProblemGenerator
# ---------------------------------------------------------------------------


def test_generator_produces_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ProblemGenerator([LinearSystem2x2Template()], split_ratio=0.8, seed=0)
        train, test = gen.generate(n_per_template=5, output_dir=tmpdir)

        assert len(train) == 4  # floor(5 * 0.8) = 4
        assert len(test) == 1

        for path, sol_path in train:
            assert os.path.exists(path)
            assert os.path.exists(sol_path)


def test_generator_solutions_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ProblemGenerator([LinearSystem2x2Template()], split_ratio=1.0, seed=7)
        train, _ = gen.generate(n_per_template=3, output_dir=tmpdir)

        for path, sol_path in train:
            graph = load_graph(path)
            sym = " ".join(v.id for v in graph.variables)
            cas = CASEngine(path, sym)
            with open(sol_path) as f:
                sol = json.load(f)
            x_vals = [sol[v.id] for v in graph.variables]
            assert cas.accepts(x_vals, tol=1e-4), f"Invalid solution in {path}"


def test_generator_multiple_templates():
    with tempfile.TemporaryDirectory() as tmpdir:
        templates = [LinearSystem2x2Template(), LinearSystem3x3Template()]
        gen = ProblemGenerator(templates, split_ratio=0.8, seed=99)
        train, test = gen.generate(n_per_template=5, output_dir=tmpdir)

        # Each template: 4 train, 1 test → totals 8 train, 2 test
        assert len(train) == 8
        assert len(test) == 2


def test_generator_deterministic():
    """Same seed produces identical file contents."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        gen1 = ProblemGenerator([LinearSystem2x2Template()], seed=5)
        gen2 = ProblemGenerator([LinearSystem2x2Template()], seed=5)
        train1, _ = gen1.generate(n_per_template=2, output_dir=tmpdir1)
        train2, _ = gen2.generate(n_per_template=2, output_dir=tmpdir2)

        for (p1, s1), (p2, s2) in zip(train1, train2):
            with open(p1) as f1, open(p2) as f2:
                assert json.load(f1) == json.load(f2)
            with open(s1) as f1, open(s2) as f2:
                assert json.load(f1) == json.load(f2)
