"""Tests for marc.structure.policy (U5 trained invention policy) and the
skip path of scripts/run_invention_eval.py."""

import json
import os
import subprocess
import sys

import pytest
import torch

from marc.structure.diffusion import corrupt, keep_schedule
from marc.structure.invention_data import make_dataset, to_padded
from marc.structure.policy import (
    SLOT_FEATURE_DIM,
    StructurePolicy,
    chosen_candidate,
    predicted_pin,
    reverse_sample,
    slot_features,
)
from marc.structure.schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph, SlotType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one(seed=0, K=4):
    [inst] = make_dataset("toys", 1, seed, K=K)
    return inst


class _Stub:
    """Fixed-output policy: strongly favors gold slots (or all-ABSENT if gold None)."""

    def __init__(self, K, gold_idx=None):
        self.K = K
        self.gold = gold_idx

    def __call__(self, inst, padded, t, T):
        n = 2 * self.K
        logits = torch.full((n, NUM_SLOT_TYPES), -10.0)
        logits[:, ABSENT] = 10.0
        vp = torch.zeros(n, 1)
        if self.gold is not None:
            g = self.gold
            logits[2 * g, :] = -10.0
            logits[2 * g, int(SlotType.VARIABLE)] = 10.0
            logits[2 * g + 1, :] = -10.0
            logits[2 * g + 1, int(SlotType.FACTOR)] = 10.0
            vp[2 * g, 0] = 1.25
            vp[2 * g + 1, 0] = -2.5
        return vp, logits


def test_slot_features_shape():
    inst = _one()
    p = to_padded(inst)
    feats = slot_features(inst, p, 5, 20)
    assert feats.shape == (2 * len(inst.candidates), SLOT_FEATURE_DIM)
    assert torch.isfinite(feats).all()


def test_policy_shapes_and_finite_grads():
    inst = _one(seed=1)
    torch.manual_seed(0)
    policy = StructurePolicy(D=32, L=1, K=4)
    clean = to_padded(inst)
    noised = corrupt(clean, 5, 10, generator=torch.Generator().manual_seed(0))
    value_pred, slot_logits = policy(inst, noised, 5, 10)
    assert value_pred.shape == (8, 1)
    assert slot_logits.shape == (8, NUM_SLOT_TYPES)
    loss = policy.head.structure_loss(slot_logits, clean.slot_types) + 0.1 * (
        policy.head.value_loss(value_pred, clean.values.unsqueeze(-1), clean.slot_types)
    )
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("single_shot", [False, True])
def test_stub_gold_recovered(single_shot):
    inst = _one(seed=2)
    K = len(inst.candidates)
    stub = _Stub(K, inst.gold_idx)
    final, logits = reverse_sample(
        stub, inst, T=10,
        generator=torch.Generator().manual_seed(0), single_shot=single_shot,
    )
    g = inst.gold_idx
    assert int(final.slot_types[2 * g]) == int(SlotType.VARIABLE)
    assert int(final.slot_types[2 * g + 1]) == int(SlotType.FACTOR)
    assert chosen_candidate(final, logits, K) == g
    # values of committed active slots come from value_pred
    assert float(final.values[2 * g]) == pytest.approx(1.25)
    assert float(final.values[2 * g + 1]) == pytest.approx(-2.5)
    # everything else stays ABSENT with zero value
    assert final.num_active() == 2


@pytest.mark.parametrize("single_shot", [False, True])
def test_stub_all_absent_gives_none(single_shot):
    inst = _one(seed=3)
    K = len(inst.candidates)
    final, logits = reverse_sample(
        _Stub(K, None), inst, T=10,
        generator=torch.Generator().manual_seed(0), single_shot=single_shot,
    )
    assert final.num_active() == 0
    assert chosen_candidate(final, logits, K) is None


def test_seeded_sampler_determinism():
    inst = _one(seed=4)
    torch.manual_seed(0)
    policy = StructurePolicy(D=16, L=1, K=4)
    policy.eval()
    f1, l1 = reverse_sample(policy, inst, T=10, generator=torch.Generator().manual_seed(7))
    f2, l2 = reverse_sample(policy, inst, T=10, generator=torch.Generator().manual_seed(7))
    assert torch.equal(f1.slot_types, f2.slot_types)
    assert torch.allclose(f1.values, f2.values)
    assert torch.allclose(l1, l2)


def test_single_shot_is_argmax_of_one_forward():
    inst = _one(seed=5)
    torch.manual_seed(0)
    policy = StructurePolicy(D=16, L=1, K=4)
    policy.eval()
    T = 10
    final, logits = reverse_sample(policy, inst, T=T, single_shot=True)
    n = 2 * len(inst.candidates)
    prior = PaddedGraph(
        torch.full((n,), ABSENT, dtype=torch.long), torch.zeros(n)
    )
    with torch.no_grad():
        _vp, ref_logits = policy(inst, prior, T - 1, T)
    assert torch.equal(final.slot_types, ref_logits.argmax(dim=-1))
    assert torch.allclose(logits, ref_logits)


def test_tiny_end_to_end_fit():
    """A tiny policy trained briefly beats 2x chance on train single-shot accuracy."""
    torch.manual_seed(0)
    K, T = 3, 10
    train = make_dataset("toys", 24, 0, K=K)
    policy = StructurePolicy(D=32, L=2, K=K)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    schedule = keep_schedule(T)
    policy.train()
    for step in range(1500):
        inst = train[step % len(train)]
        clean = to_padded(inst)
        t = int(torch.randint(1, T, (1,), generator=gen).item())
        noised = corrupt(clean, t, T, schedule=schedule, generator=gen)
        vp, logits = policy(inst, noised, t, T)
        loss = policy.head.structure_loss(logits, clean.slot_types) + 0.1 * (
            policy.head.value_loss(vp, clean.values.unsqueeze(-1), clean.slot_types)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval()
    hits = 0
    for inst in train:
        final, logits = reverse_sample(policy, inst, T=T, single_shot=True)
        if chosen_candidate(final, logits, K) == inst.gold_idx:
            hits += 1
    acc = hits / len(train)
    chance = 1.0 / (K + 1)
    assert acc > 2 * chance, f"train single-shot invention acc {acc:.3f} <= 2x chance"


# --- context ablation + predicted-pin readout ---------------------------------


def test_ablate_context_default_false_and_attribute():
    policy = StructurePolicy(D=16, L=1, K=2)
    assert policy.ablate_context is False
    assert policy.encoder.ablate_context is False
    ablated = StructurePolicy(D=16, L=1, K=2, ablate_context=True)
    assert ablated.ablate_context is True
    assert ablated.encoder.ablate_context is True


def test_ablate_state_dict_round_trip():
    """Ablation changes zero parameters: strict load works in both directions,
    including from a trained (stepped) normal checkpoint."""
    torch.manual_seed(0)
    inst = _one(seed=6)
    normal = StructurePolicy(D=16, L=1, K=4)
    # one real training step so the checkpoint isn't just init state
    clean = to_padded(inst)
    noised = corrupt(clean, 3, 10, generator=torch.Generator().manual_seed(0))
    vp, logits = normal(inst, noised, 3, 10)
    loss = normal.head.structure_loss(logits, clean.slot_types)
    loss.backward()
    with torch.no_grad():
        for p in normal.parameters():
            if p.grad is not None:
                p -= 1e-2 * p.grad
    ablated = StructurePolicy(D=16, L=1, K=4, ablate_context=True)
    ablated.load_state_dict(normal.state_dict())  # strict by default
    normal.load_state_dict(ablated.state_dict())


def test_ablated_output_invariant_to_fixed_graph_constants():
    """Perturbing the fixed graph's factor constants must move the normal
    policy's logits but leave the ablated policy's logits bit-identical —
    the direct measurement that ctx is the only fixed-graph pathway."""
    import dataclasses

    from marc.graph.schema import FactorNode

    torch.manual_seed(0)
    inst = _one(seed=7)
    perturbed_graph = dataclasses.replace(
        inst.fixed_graph,
        factors=[FactorNode(f.id, f"({f.expression}) + 0.5")
                 for f in inst.fixed_graph.factors],
    )
    # new id so StructurePolicy's HeteroData cache can't serve the stale graph
    pert = dataclasses.replace(inst, id=inst.id + "_pert", fixed_graph=perturbed_graph)

    normal = StructurePolicy(D=16, L=1, K=4)
    ablated = StructurePolicy(D=16, L=1, K=4, ablate_context=True)
    ablated.load_state_dict(normal.state_dict())  # same weights, only ctx zeroed
    normal.eval()
    ablated.eval()

    padded = corrupt(to_padded(inst), 4, 10, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        _, ln1 = normal(inst, padded, 4, 10)
        _, ln2 = normal(pert, padded, 4, 10)
        _, la1 = ablated(inst, padded, 4, 10)
        _, la2 = ablated(pert, padded, 4, 10)
    assert not torch.allclose(ln1, ln2), "normal policy ignored the fixed graph"
    assert torch.equal(la1, la2), "ablated policy still reads the fixed graph"


def test_predicted_pin_reads_factor_slot_and_none_when_absent():
    K = 3
    types = torch.full((2 * K,), ABSENT, dtype=torch.long)
    values = torch.zeros(2 * K)
    types[2] = int(SlotType.VARIABLE)
    types[3] = int(SlotType.FACTOR)
    values[2] = 1.25
    values[3] = -2.5
    final = PaddedGraph(types, values)
    assert predicted_pin(final, 1) == pytest.approx(-2.5)
    assert predicted_pin(final, 0) is None
    assert predicted_pin(final, 2) is None


def test_predicted_pin_from_reverse_sample_stub():
    inst = _one(seed=8)
    K = len(inst.candidates)
    stub = _Stub(K, inst.gold_idx)
    final, _ = reverse_sample(stub, inst, T=10,
                              generator=torch.Generator().manual_seed(0))
    g = inst.gold_idx
    assert predicted_pin(final, g) == pytest.approx(-2.5)
    assert all(predicted_pin(final, j) is None for j in range(K) if j != g)


# --- eval script skip path ----------------------------------------------------


def test_eval_skipped_path(tmp_path):
    out = tmp_path / "skip.json"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_invention_eval.py"),
         "--ckpt", str(tmp_path / "nonexistent.pt"), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["status"] == "skipped"
    assert "reason" in data
