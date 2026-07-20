"""Procedural aux-required family (marc/data/aux_required.py).

Pins the H2 pairing at dataset scale: every generated fixed graph is certified
inconsistent, every augmented graph certified uniquely solvable with the gold
accepted by the Checker, and (fixed + insert_coeffs + defining factor) exactly
reconstructs the augmented graph — the recipe a structure policy must learn.
"""

from functools import lru_cache

import pytest
import sympy as sp

from marc.cas.checker import Checker
from marc.data.aux_required import (
    AUX_REQUIRED_TEMPLATES,
    PATTERNS,
    AuxRequiredTemplate,
    generate_instances,
    verify_instance,
)
from marc.data.generator import ProblemGenerator
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode


@lru_cache(maxsize=None)
def _inst(pattern, seed):
    # ponytail: cached — generation re-certifies with sympy; tests never mutate.
    return AuxRequiredTemplate(pattern).generate_instance(seed)


def _instances(pattern, n=10):
    return [_inst(pattern, seed) for seed in range(n)]


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_verify_instance_passes(pattern):
    for inst in _instances(pattern):
        assert verify_instance(inst), f"{inst.id} failed a certificate"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_checker_rejects_gold_projection_on_fixed(pattern):
    checker = Checker()
    for inst in _instances(pattern):
        base_gold = [inst.solution[v.id] for v in inst.fixed_graph.variables]
        assert not checker.verify(inst.fixed_graph, base_gold).accepted, (
            f"{inst.id}: fixed graph should reject the base projection of gold"
        )


# ---------------------------------------------------------------------------
# structure: augmented = fixed + one latent + one defining factor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_augmented_extends_fixed_by_one(pattern):
    for inst in _instances(pattern):
        assert len(inst.augmented_graph.variables) == len(inst.fixed_graph.variables) + 1
        assert len(inst.augmented_graph.factors) == len(inst.fixed_graph.factors) + 1
        assert inst.aux_var not in {v.id for v in inst.fixed_graph.variables}
        u = sp.Symbol(inst.aux_var)
        for f in inst.fixed_graph.factors:
            assert u not in sp.sympify(f.expression).free_symbols, (
                f"{inst.id}: {f.id} still references {inst.aux_var}"
            )


@pytest.mark.parametrize("pattern", PATTERNS)
def test_defining_expression_zero_at_gold(pattern):
    for inst in _instances(pattern):
        expr = sp.sympify(inst.defining_expression)
        assert expr.subs(sp.Symbol(inst.aux_var), inst.solution[inst.aux_var]) == 0
        assert inst.solution[inst.aux_var] == inst.aux_value
        assert inst.aux_value != 0  # dropping the latent must matter


def _rebuild_augmented(inst):
    """fixed + insert_coeffs + defining factor -> augmented (the C3 recipe)."""
    variables = [VariableNode(v.id, v.value) for v in inst.fixed_graph.variables]
    variables.append(VariableNode(inst.aux_var, 0.0))
    factors = []
    edges = [Edge(e.variable_id, e.factor_id, e.coefficient)
             for e in inst.fixed_graph.edges]
    for f in inst.fixed_graph.factors:
        coef = inst.insert_coeffs.get(f.id)
        if coef is None:
            factors.append(FactorNode(f.id, f.expression))
        else:
            factors.append(FactorNode(f.id, f"{f.expression}+({coef})*{inst.aux_var}"))
            edges.append(Edge(inst.aux_var, f.id, coef))
    factors.append(FactorNode("aux", inst.defining_expression))
    edges.append(Edge(inst.aux_var, "aux", 1.0))
    return FactorGraph(variables, factors, edges)


def _graphs_equal(g1, g2):
    """Order-insensitive equality: var ids, per-id sympy factor equality, edge sets."""
    if {v.id for v in g1.variables} != {v.id for v in g2.variables}:
        return False
    f1 = {f.id: sp.sympify(f.expression) for f in g1.factors}
    f2 = {f.id: sp.sympify(f.expression) for f in g2.factors}
    if f1.keys() != f2.keys():
        return False
    if any(sp.simplify(f1[k] - f2[k]) != 0 for k in f1):
        return False
    e1 = {(e.variable_id, e.factor_id, float(e.coefficient)) for e in g1.edges}
    e2 = {(e.variable_id, e.factor_id, float(e.coefficient)) for e in g2.edges}
    return e1 == e2


@pytest.mark.parametrize("pattern", PATTERNS)
def test_insert_coeffs_reconstruct_augmented(pattern):
    for inst in _instances(pattern):
        assert _graphs_equal(_rebuild_augmented(inst), inst.augmented_graph), (
            f"{inst.id}: fixed + insert_coeffs + defining factor != augmented"
        )


def test_shape_regression():
    expected = {"offset": (3, 4), "coupled": (4, 5), "shared": (3, 4)}
    for pattern, (n_vars, n_factors) in expected.items():
        graph, sol = AuxRequiredTemplate(pattern).generate(seed=0)
        assert len(graph.variables) == n_vars
        assert len(graph.factors) == n_factors
        assert set(sol) == {v.id for v in graph.variables}


# ---------------------------------------------------------------------------
# determinism / diversity / API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_deterministic_and_diverse(pattern):
    tmpl = AuxRequiredTemplate(pattern)
    assert tmpl.generate_instance(3) == tmpl.generate_instance(3)
    golds = {tuple(sorted(inst.solution.items())) for inst in _instances(pattern)}
    assert len(golds) >= 2


def test_generate_instances_round_robin():
    insts = generate_instances(6, seed=10)
    assert [i.pattern for i in insts] == PATTERNS * 2
    assert insts[0].id == f"{PATTERNS[0]}_10"
    assert [i.seed for i in insts] == list(range(10, 16))
    only = generate_instances(2, seed=0, patterns=["shared"])
    assert [i.pattern for i in only] == ["shared", "shared"]


def test_templates_export():
    assert [t.pattern for t in AUX_REQUIRED_TEMPLATES] == PATTERNS
    assert AUX_REQUIRED_TEMPLATES[0].name == "AuxRequired_offset"


def test_problem_generator_runs_clean(tmp_path):
    train, test = ProblemGenerator([AuxRequiredTemplate("offset")], seed=1).generate(2, tmp_path)
    assert len(train) + len(test) == 2
