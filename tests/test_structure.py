"""Tests for the P3 structure-diffusion prototype (marc/structure/*)."""

import torch

from marc.model.structure_head import StructureHead
from marc.structure.diffusion import (
    absent_fraction,
    corrupt,
    corrupt_types,
    keep_schedule,
)
from marc.structure.schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph, SlotType


# --- schema ----------------------------------------------------------------

def test_absent_is_index_zero_matches_structure_head():
    assert ABSENT == 0
    assert ABSENT == StructureHead.ABSENT_TYPE


def test_from_active_pads_and_masks():
    g = PaddedGraph.from_active([2.0, 1.0, 3.0], n_slots=8)
    assert g.n_slots == 8
    assert g.num_active() == 3
    assert g.active_mask().tolist() == [True, True, True, False, False, False, False, False]
    # ABSENT slots carry zero value
    assert g.values[3:].abs().sum().item() == 0.0


def test_from_active_rejects_overflow():
    try:
        PaddedGraph.from_active([1.0] * 5, n_slots=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when values exceed slot count")


def test_to_features_shape_and_onehot():
    g = PaddedGraph.from_active([2.0, 1.0], n_slots=4)
    feats = g.to_features()
    assert feats.shape == (4, NUM_SLOT_TYPES + 1)
    # one-hot part of an ABSENT slot points at index 0
    assert feats[2, ABSENT].item() == 1.0
    # value column zero for ABSENT slot
    assert feats[2, -1].item() == 0.0


# --- forward corruption ----------------------------------------------------

def test_keep_schedule_endpoints():
    sched = keep_schedule(50)
    assert sched[0].item() == 1.0            # t=0 -> identity
    assert sched[-1].item() < 1e-6           # t=T-1 -> fully absorbed


def test_corrupt_types_identity_at_t0():
    types = torch.tensor([1, 1, 0, 1])
    out = corrupt_types(types, t=0, T=50)
    assert torch.equal(out, types)           # keep-prob 1.0 -> unchanged


def test_corrupt_types_absorbs_toward_absent_at_high_t():
    types = torch.ones(200, dtype=torch.long)  # all active
    out = corrupt_types(types, t=49, T=50, generator=torch.Generator().manual_seed(0))
    # near t=T-1 keep-prob ~0, so almost everything is ABSENT and nothing new is invented
    assert absent_fraction(out) > 0.9
    assert set(out.tolist()).issubset(set(range(NUM_SLOT_TYPES)))


def test_corruption_is_monotone_in_expectation():
    types = torch.ones(500, dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    low = absent_fraction(corrupt_types(types, t=5, T=50, generator=gen))
    high = absent_fraction(corrupt_types(types, t=45, T=50, generator=gen))
    assert high > low


def test_corrupt_graph_zeros_absent_values():
    g = PaddedGraph.from_active([2.0, 1.0, 3.0], n_slots=6)
    noised = corrupt(g, t=49, T=50, generator=torch.Generator().manual_seed(0))
    # every ABSENT slot must have value 0
    absent = noised.slot_types == ABSENT
    assert noised.values[absent].abs().sum().item() == 0.0


# --- integration with the reverse head -------------------------------------

def test_structure_head_consumes_corrupted_graph():
    torch.manual_seed(0)
    g = PaddedGraph.from_active([2.0, 1.0, 3.0], n_slots=8)
    noised = corrupt(g, t=30, T=50, generator=torch.Generator().manual_seed(0))
    D = 16
    encoder = torch.nn.Linear(NUM_SLOT_TYPES + 1, D)
    head = StructureHead(D=D, num_slot_types=NUM_SLOT_TYPES)
    eps_hat, logits = head(encoder(noised.to_features()))
    assert eps_hat.shape == (8, 1)
    assert logits.shape == (8, NUM_SLOT_TYPES)
    loss = head.structure_loss(logits, g.slot_types)
    assert loss.requires_grad and loss.item() >= 0


def test_pilot_recovers_and_flips_absent_to_active():
    """Short end-to-end pilot: training reduces loss and a slot flips ABSENT->active."""
    from scripts.train_structure_pilot import train

    res = train(steps=300, seed=0)
    assert res["loss_end"] < res["loss_start"]        # learning happened
    assert len(res["transitions"]) > 0                # ABSENT -> active occurred
