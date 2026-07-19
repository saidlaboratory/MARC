"""Tests for the P4 coordinate-geometry templates (marc/data/templates.py).

Done-condition: geometry samples generate and pass the CAS check. We verify each
generated instance with BOTH the numeric CASEngine (as the generator does) and the
conservative exact Checker, at the stored integer solution x*.
"""

import os
import tempfile

import pytest

from marc.cas.checker import Checker
from marc.cas.engine import CASEngine
from marc.data.templates import (
    GEOMETRY_TEMPLATES,
    PointSlopeTemplate,
    TriangleDistanceTemplate,
)
from marc.graph.serialize import save_graph

N_SAMPLES = 20


def _ordered_solution(graph, solution):
    """Solution dict -> value list in the graph's variable order."""
    return [solution[v.id] for v in graph.variables]


def _cas_accepts(graph, solution) -> bool:
    """Replicate the generator's CAS invariant check without touching disk state."""
    symbol_names = " ".join(v.id for v in graph.variables)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        save_graph(graph, path)
        cas = CASEngine(path, symbol_names)
        return cas.accepts(_ordered_solution(graph, solution), tol=1e-4)
    finally:
        os.unlink(path)


@pytest.mark.parametrize("template", GEOMETRY_TEMPLATES, ids=lambda t: t.name)
def test_geometry_samples_pass_cas_check(template):
    """Spec: generate 20 samples; CAS accepts each at x*."""
    checker = Checker()
    for i in range(N_SAMPLES):
        graph, solution = template.generate(seed=i)
        x = _ordered_solution(graph, solution)
        # numeric CAS gate (matches ProblemGenerator's invariant)
        assert _cas_accepts(graph, solution), f"{template.name} #{i}: CAS rejected x*"
        # conservative exact/symbolic gate
        assert checker.verify(graph, x).accepted, f"{template.name} #{i}: checker rejected x*"


def test_points_are_variables_and_relations_are_factors():
    """Structural sanity: even variable count (coordinate pairs) + a nonlinear factor."""
    graph, _ = TriangleDistanceTemplate().generate(seed=0)
    assert len(graph.variables) % 2 == 0
    assert any("**2" in f.expression for f in graph.factors)  # a genuine distance factor


def test_point_slope_has_a_slope_factor():
    graph, _ = PointSlopeTemplate().generate(seed=0)
    assert any(f.id == "slope_ap" for f in graph.factors)
    # and a distance factor too
    assert any("**2" in f.expression for f in graph.factors)


def test_perturbing_the_solution_is_rejected():
    """A wrong assignment must fail the checker (guards against trivially-true factors)."""
    checker = Checker()
    graph, solution = TriangleDistanceTemplate().generate(seed=3)
    x = _ordered_solution(graph, solution)
    x_bad = list(x)
    x_bad[-1] += 1.0  # nudge cy off the solution
    assert not checker.verify(graph, x_bad).accepted


def test_generator_accepts_geometry_templates(tmp_path):
    """End-to-end: ProblemGenerator's built-in CAS assertion passes on geometry."""
    from marc.data.generator import ProblemGenerator

    gen = ProblemGenerator(GEOMETRY_TEMPLATES, split_ratio=0.8, seed=7)
    train, test = gen.generate(n_per_template=N_SAMPLES, output_dir=str(tmp_path))
    # 2 templates x 20 samples, 80/20 split
    assert len(train) + len(test) == 2 * N_SAMPLES
