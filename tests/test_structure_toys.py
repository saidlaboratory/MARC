"""H2 structure toys (marc/eval/structure_toys.py): the checker is ground truth.

The claim these toys encode: an augmented graph (base vars + one latent) is
consistent and uniquely solvable, while the fixed graph (latent removed) is
over-determined and inconsistent — so a fixed-structure solver can never satisfy
every factor. These tests pin that claim to the Checker directly (no solver, no
randomness).
"""

from marc.cas.checker import Checker
from marc.eval.structure_toys import (
    all_structure_toys,
    structure_toys_augmented,
    structure_toys_fixed,
)


def test_toy_counts():
    assert len(structure_toys_fixed()) == 3
    assert len(structure_toys_augmented()) == 3
    assert len(all_structure_toys()) == 6


def test_augmented_toys_accept_gold_solution():
    checker = Checker()
    for p in structure_toys_augmented():
        result = checker.verify(p.graph, p.solution)
        assert result.accepted, f"{p.id} should accept its gold solution"
        assert result.max_residual < 1e-6


def test_fixed_toys_are_over_determined_and_rejected():
    checker = Checker()
    for p in structure_toys_fixed():
        # Removing the latent leaves more constraints than unknowns...
        assert len(p.graph.factors) > len(p.graph.variables), (
            f"{p.id} should be over-determined"
        )
        # ...and no assignment (including the stored gold) satisfies all of them.
        assert not checker.verify(p.graph, p.solution).accepted


def test_augmented_adds_exactly_one_variable_over_fixed():
    fixed = {p.id.replace("_fixed", ""): p for p in structure_toys_fixed()}
    aug = {p.id.replace("_augmented", ""): p for p in structure_toys_augmented()}
    for key in fixed:
        assert len(aug[key].graph.variables) == len(fixed[key].graph.variables) + 1
