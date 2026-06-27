"""Tests for marc/eval/problems.py — generators with known-exact solutions."""

import pytest

from marc.cas.checker import Checker
from marc.eval.problems import entrapment_suite, held_out_structure, in_distribution


@pytest.mark.parametrize(
    "family,n,split,n_vars",
    [
        (in_distribution, 25, "in_distribution", 2),
        (held_out_structure, 25, "held_out_structure", 3),
        (entrapment_suite, 50, "entrapment", 1),
    ],
)
def test_known_solutions_are_accepted(family, n, split, n_vars):
    checker = Checker()
    problems = family(n)
    assert len(problems) == n
    for p in problems:
        assert p.metadata["split"] == split
        assert len(p.graph.variables) == n_vars
        assert len(p.solution) == n_vars
        # the stored solution must be an exact root the checker accepts
        assert checker.verify(p.graph, p.solution).accepted, p.id


def test_generators_are_deterministic():
    a = in_distribution(10, seed=3)
    b = in_distribution(10, seed=3)
    assert [p.solution for p in a] == [p.solution for p in b]


def test_entrapment_starts_are_off_solution():
    # the seeded start must not already sit on the solution (else nothing to escape)
    for p in entrapment_suite(50):
        assert abs(p.metadata["init"][0] - p.solution[0]) > 0.5
