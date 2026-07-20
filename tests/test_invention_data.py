"""Tests for marc.structure.invention_data (U5 menu-based invention data)."""

import sys

import pytest
import torch

from marc.cas.checker import Checker
from marc.structure.diffusion import absent_fraction, corrupt
from marc.structure.invention_data import (
    DATA_VERSION,
    DEFAULT_PROBE,
    FAMILIES,
    FAMILIES_BY_SOURCE,
    REFERENCE_SOLVER,
    SOURCES,
    Candidate,
    _candidate_key,
    _is_inconsistent,
    _solvable_at_eval_grade,
    certify_unsolvable,
    make_dataset,
    to_padded,
)
from marc.structure.schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph, SlotType


def test_slot_type_vocab_extended():
    assert int(SlotType.FACTOR) == 2
    assert NUM_SLOT_TYPES == 3
    # feature width follows the vocabulary size
    g = PaddedGraph.from_active([1.0, 2.0], n_slots=4)
    assert g.to_features().shape == (4, NUM_SLOT_TYPES + 1)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(5))
def test_menu_certified(family, seed):
    [inst] = make_dataset("toys", 1, seed, K=4, families=[family])
    assert inst.family == family
    assert len(inst.candidates) == 4
    assert _is_inconsistent(inst.fixed_graph)

    # gold apply -> checker accepts the stored augmented solution
    gold = inst.candidates[inst.gold_idx]
    aug = gold.apply(inst.fixed_graph)
    x = [inst.solution[v.id] for v in aug.variables]
    assert Checker().accepts(aug, x)

    # every distractor certified inconsistent; exactly one solvable option
    solvable = 0
    for j, cand in enumerate(inst.candidates):
        applied = cand.apply(inst.fixed_graph)
        if _is_inconsistent(applied):
            assert j != inst.gold_idx
        else:
            solvable += 1
            assert j == inst.gold_idx
    assert solvable == 1

    # hard negative: shares gold insert_coeffs, differs in pin, inconsistent
    hard = [
        c for j, c in enumerate(inst.candidates)
        if j != inst.gold_idx and c.insert_coeffs == gold.insert_coeffs
    ]
    assert len(hard) >= 1
    assert all(h.pin_value != gold.pin_value for h in hard)
    assert all(_is_inconsistent(h.apply(inst.fixed_graph)) for h in hard)


def test_gold_apply_is_pure():
    [inst] = make_dataset("toys", 1, 0, K=4)
    before = [(f.id, f.expression) for f in inst.fixed_graph.factors]
    n_vars, n_edges = len(inst.fixed_graph.variables), len(inst.fixed_graph.edges)
    inst.candidates[inst.gold_idx].apply(inst.fixed_graph)
    assert [(f.id, f.expression) for f in inst.fixed_graph.factors] == before
    assert len(inst.fixed_graph.variables) == n_vars
    assert len(inst.fixed_graph.edges) == n_edges


def test_no_hard_negatives_flag():
    [inst] = make_dataset("toys", 1, 0, K=4, hard_negatives=False)
    gold = inst.candidates[inst.gold_idx]
    # without the flag no distractor is required to share the gold structure
    assert len(inst.candidates) == 4
    for j, c in enumerate(inst.candidates):
        if j != inst.gold_idx:
            assert _is_inconsistent(c.apply(inst.fixed_graph))
    assert gold.aux_var == "u"


def test_deterministic_and_gold_idx_varies():
    ds1 = make_dataset("toys", 12, 0, K=4)
    ds2 = make_dataset("toys", 12, 0, K=4)
    assert [i.id for i in ds1] == [i.id for i in ds2]
    assert [i.gold_idx for i in ds1] == [i.gold_idx for i in ds2]
    assert [
        (i.candidates[i.gold_idx].pin_value, i.candidates[i.gold_idx].insert_coeffs)
        for i in ds1
    ] == [
        (i.candidates[i.gold_idx].pin_value, i.candidates[i.gold_idx].insert_coeffs)
        for i in ds2
    ]
    # gold position is rng-shuffled per instance seed
    assert len({i.gold_idx for i in ds1}) > 1


def test_to_padded_and_corruption():
    [inst] = make_dataset("toys", 1, 3, K=4)
    p = to_padded(inst)
    assert p.n_slots == 8
    assert p.num_active() == 2
    g = inst.gold_idx
    assert int(p.slot_types[2 * g]) == int(SlotType.VARIABLE)
    assert int(p.slot_types[2 * g + 1]) == int(SlotType.FACTOR)
    assert float(p.values[2 * g]) == inst.aux_value
    assert float(p.values[2 * g + 1]) == inst.candidates[g].pin_value
    # all other slots ABSENT with value 0
    for j in range(8):
        if j not in (2 * g, 2 * g + 1):
            assert int(p.slot_types[j]) == ABSENT
            assert float(p.values[j]) == 0.0

    # corrupt at t = T-1 leaves most slots ABSENT (cosine keep-prob ~ 0)
    T = 20
    gen = torch.Generator().manual_seed(0)
    noised = corrupt(p, T - 1, T, generator=gen)
    assert absent_fraction(noised.slot_types) >= 0.75


def test_aux_required_missing_raises_clean_error(monkeypatch):
    # simulate the module being absent regardless of what this branch carries
    monkeypatch.setitem(sys.modules, "marc.data.aux_required", None)
    # a prior import in the same session binds marc.data.aux_required as a package
    # attribute, which the from-import consults before sys.modules — drop it too
    # so this test is order-independent in the full suite (PR #65 fix)
    import marc.data

    monkeypatch.delattr(marc.data, "aux_required", raising=False)
    with pytest.raises(ImportError, match="aux_required"):
        make_dataset("aux_required", 1, 0)


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        make_dataset("nope", 1, 0)


# --- W2: defining expressions, certificates, support randomization, nonlinear ---


def test_module_constants():
    assert SOURCES == ("toys", "aux_required", "nonlinear")
    assert FAMILIES_BY_SOURCE["toys"] == FAMILIES
    assert FAMILIES_BY_SOURCE["aux_required"] == FAMILIES
    assert FAMILIES_BY_SOURCE["nonlinear"] == ("vieta", "quad_link")
    assert DATA_VERSION == 2


def test_reference_solver_literal():
    # must match run_invention_eval.py / train_structure_policy.py's solver config
    assert REFERENCE_SOLVER == {"name": "refine", "k_refine": 4, "polish_steps": 4000}


def test_defining_expression_apply():
    [inst] = make_dataset("toys", 1, 0, K=4)  # toy1: variables x, y
    gold = inst.candidates[inst.gold_idx]

    # scalar-pin path (defining_expression=None) keeps today's template
    aug0 = gold.apply(inst.fixed_graph)
    aux0 = next(f for f in aug0.factors if f.id == "aux")
    assert aux0.expression == f"{gold.aux_var} - ({gold.pin_value})"

    cand = Candidate(gold.aux_var, gold.pin_value, dict(gold.insert_coeffs), "u - (x - y)")
    aug = cand.apply(inst.fixed_graph)
    aux = next(f for f in aug.factors if f.id == "aux")
    assert aux.expression == "u - (x - y)"
    # edges cover the aux var and every fixed-graph variable in the expression
    assert {e.variable_id for e in aug.edges if e.factor_id == "aux"} == {"u", "x", "y"}
    # pin_value retained for features/back-compat
    assert cand.pin_value == gold.pin_value
    # the key distinguishes expression candidates from scalar-pin ones
    assert _candidate_key(cand) != _candidate_key(gold)
    assert _candidate_key(cand) != _candidate_key(
        Candidate(gold.aux_var, gold.pin_value, dict(gold.insert_coeffs), "u - (x + y)")
    )
    # aux_var must appear in the expression
    with pytest.raises(ValueError, match="aux_var"):
        Candidate("u", 0.0, {}, "x - y")


@pytest.mark.parametrize("family", FAMILIES)
def test_gold_support_varies(family):
    supports = set()
    for seed in range(10):
        [inst] = make_dataset("toys", 1, seed, K=4, families=[family])
        supports.add(frozenset(inst.candidates[inst.gold_idx].insert_coeffs))
    # support randomization: the gold's touched-factor set is not family-constant
    assert len(supports) > 1


def test_linear_sources_exact_certificate():
    ds = make_dataset("toys", 3, 0, K=4)
    assert all(i.certificate == "exact" for i in ds)
    assert all(i.certificate_config is None for i in ds)
    # _is_inconsistent still importable and behaves on linear graphs
    assert _is_inconsistent(ds[0].fixed_graph)
    assert certify_unsolvable(ds[0].fixed_graph, rng_seed=0) == {
        "unsolvable": True,
        "method": "linear_rank",
    }


def test_nonlinear_menu_end_to_end():
    # THE expensive test (~30-60s): 2 instances, one per nonlinear family
    ds = make_dataset("nonlinear", 2, 0, K=4)
    assert [i.family for i in ds] == ["vieta", "quad_link"]
    for inst in ds:
        assert inst.certificate == "empirical"
        cfg = inst.certificate_config
        assert {k: cfg[k] for k in DEFAULT_PROBE} == DEFAULT_PROBE
        assert "empirical claim" in cfg["claim"]

        gold = inst.candidates[inst.gold_idx]
        assert gold.defining_expression is not None
        aug = gold.apply(inst.fixed_graph)
        # stored solution satisfies the augmented graph exactly
        x = [inst.solution[v.id] for v in aug.variables]
        assert Checker().accepts(aug, x)
        # gold-applied solves at eval grade (the reference solver + Checker)
        assert _solvable_at_eval_grade(aug, rng_seed=123)
        # the fixed graph fails the probe (empirically unsolvable)
        assert certify_unsolvable(inst.fixed_graph, rng_seed=7) == {
            "unsolvable": True,
            "method": "empirical_probe",
        }
        # hard negative: shares gold coeffs + expression shape, perturbed constant
        # (filter on the expression: a scalar-pin filler may coincidentally share
        # the gold insert_coeffs, but only the hard negative carries an expression)
        hard = [
            c for j, c in enumerate(inst.candidates)
            if j != inst.gold_idx
            and c.insert_coeffs == gold.insert_coeffs
            and c.defining_expression is not None
        ]
        assert len(hard) == 1
        assert hard[0].defining_expression.startswith(gold.defining_expression)
        assert hard[0].defining_expression != gold.defining_expression
