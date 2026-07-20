"""Tests for marc.structure.policy (U5 trained invention policy) and the eval
JSON block builder in scripts/run_invention_eval.py."""

import importlib.util
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


# --- eval JSON block builder --------------------------------------------------


def _load_eval_module():
    path = os.path.join(ROOT, "scripts", "run_invention_eval.py")
    spec = importlib.util.spec_from_file_location("run_invention_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_block_schema():
    mod = _load_eval_module()
    torch.manual_seed(0)
    K = 3
    instances = make_dataset("toys", 4, 100, K=K)
    policy = StructurePolicy(D=16, L=1, K=K)
    policy.eval()
    res = mod.evaluate(policy, instances, k_refine=1, T=6, seed=0)
    assert set(res) == {"positive_control_ok", "samplers"}
    assert isinstance(res["positive_control_ok"], bool)
    for sampler in ("reverse", "single_shot"):
        block = res["samplers"][sampler]
        assert set(block) == {
            "invention_rate", "none_rate", "hard_negative_confusion",
            "solve", "comparisons", "per_family",
        }
        assert set(block["solve"]) == {
            "fixed", "policy", "random_slot", "always_none", "gold_oracle",
        }
        for r in [block["invention_rate"], block["none_rate"],
                  block["hard_negative_confusion"], *block["solve"].values()]:
            assert set(r) == {"k", "n", "rate", "ci95"}
        assert set(block["comparisons"]) == {"policy_vs_random", "policy_vs_fixed"}
        for cmp in block["comparisons"].values():
            assert set(cmp) == {"z", "p"}
        for fam_block in block["per_family"].values():
            assert set(fam_block) == {"invention_rate", "solve_policy"}
    # the fixed graph is inconsistent, so fixed/always_none can never solve
    assert res["samplers"]["reverse"]["solve"]["fixed"]["k"] == 0
    # the gold graph is consistent + easy: positive control should hold
    assert res["samplers"]["reverse"]["solve"]["gold_oracle"]["rate"] >= 0.95


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
