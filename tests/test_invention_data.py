"""Tests for marc.structure.invention_data (U5 menu-based invention data)."""

import sys

import pytest
import torch

from marc.cas.checker import Checker
from marc.structure.diffusion import absent_fraction, corrupt
from marc.structure.invention_data import (
    FAMILIES,
    _is_inconsistent,
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
    # simulate the module being absent regardless of what this branch carries;
    # if an earlier test already imported it, `from marc.data import aux_required`
    # resolves via the package attribute without consulting sys.modules — remove both
    import marc.data

    monkeypatch.setitem(sys.modules, "marc.data.aux_required", None)
    monkeypatch.delattr(marc.data, "aux_required", raising=False)
    with pytest.raises(ImportError, match="aux_required"):
        make_dataset("aux_required", 1, 0)


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown source"):
        make_dataset("nope", 1, 0)
