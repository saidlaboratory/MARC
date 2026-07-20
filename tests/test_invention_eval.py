"""Tests for the W1 eval protocol in scripts/run_invention_eval.py:
seed hygiene, frozen legacy evaluate(), evaluate_full schema, Holm, enumeration,
capability-guarded arms."""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from marc.structure.invention_data import make_dataset
from marc.structure.schema import ABSENT, NUM_SLOT_TYPES, SlotType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_invention_eval.py")


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("run_invention_eval_w1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_eval_module()


class _GoldStub:
    """Fixed-output policy: strongly favors each instance's gold slots."""

    def __call__(self, inst, padded, t, T):
        n = 2 * len(inst.candidates)
        logits = torch.full((n, NUM_SLOT_TYPES), -10.0)
        logits[:, ABSENT] = 10.0
        g = inst.gold_idx
        logits[2 * g, :] = -10.0
        logits[2 * g, int(SlotType.VARIABLE)] = 10.0
        logits[2 * g + 1, :] = -10.0
        logits[2 * g + 1, int(SlotType.FACTOR)] = 10.0
        return torch.zeros(n, 1), logits


# --- 1/2: seed hygiene -------------------------------------------------------


def test_overlap_assertion_fires(mod):
    train_config = {"data": "toys", "train_seed_range": [900000, 900500],
                    "val_seed_range": [500000, 500050]}
    with pytest.raises(SystemExit) as exc:
        mod._check_seed_hygiene(train_config, "toys", [900000, 901000], 4, False)
    assert exc.value.code == 2


def test_legacy_reconstruction(mod):
    # only seed/n_train present (today's train_config): ranges reconstructed.
    train_config = {"data": "toys", "seed": 0, "n_train": 500}
    hyg = mod._check_seed_hygiene(train_config, "toys", [900000, 901000], 4, False)
    assert hyg["source"] == "reconstructed"
    assert hyg["train_seed_range"] == [0, 500]
    assert hyg["val_seed_range"] == [500000, 500050]
    assert hyg["overlap_instances"] == 0
    # reconstructed ranges still DETECT overlap (val range hit by a 500xxx eval seed)
    with pytest.raises(SystemExit) as exc:
        mod._check_seed_hygiene(train_config, "toys", [500000], 4, False)
    assert exc.value.code == 2
    # --allow-seed-overlap: records the contamination instead of exiting
    hyg = mod._check_seed_hygiene(train_config, "toys", [500000], 4, True)
    assert hyg["overlap_instances"] == 4
    assert hyg["allow_seed_overlap"] is True


# --- 3: legacy evaluate() schema is frozen -----------------------------------


def test_legacy_evaluate_keys_unchanged(mod):
    instances = make_dataset("toys", 3, 100, K=3)
    res = mod.evaluate(_GoldStub(), instances, k_refine=1, T=6, seed=0)
    assert set(res) == {"positive_control_ok", "samplers"}
    for sampler in ("reverse", "single_shot"):
        block = res["samplers"][sampler]
        assert set(block) == {
            "invention_rate", "none_rate", "hard_negative_confusion",
            "solve", "comparisons", "per_family",
        }
        assert set(block["solve"]) == {
            "fixed", "policy", "random_slot", "always_none", "gold_oracle",
        }
        assert set(block["comparisons"]) == {"policy_vs_random", "policy_vs_fixed"}
        for cmp in block["comparisons"].values():
            assert set(cmp) == {"z", "p"}


# --- 4: evaluate_full schema + pooled counts ---------------------------------


def _walk_rates(block):
    """Yield every {k,n,rate,ci95}-shaped dict reachable in a nested block."""
    if isinstance(block, dict):
        if set(block) == {"k", "n", "rate", "ci95"}:
            yield block
        else:
            for v in block.values():
                yield from _walk_rates(v)
    elif isinstance(block, list):
        for v in block:
            yield from _walk_rates(v)


def test_evaluate_full_schema_and_pooling(mod):
    res = mod.evaluate_full(_GoldStub(), data="toys", n=2, K=3, k_refine=1,
                            T=6, eval_seeds=[900000, 901000], ckpt=None)
    assert set(res) == {
        "positive_control_ok", "samplers", "arms", "comparisons_holm",
        "timing", "positive_control", "per_seed", "caveats",
    }
    for r in _walk_rates({"samplers": res["samplers"], "arms": res["arms"],
                          "positive_control": res["positive_control"]}):
        assert set(r) == {"k", "n", "rate", "ci95"}
        assert 0.0 <= r["rate"] <= 1.0
    # pooled counts == sum of per-seed counts
    assert len(res["per_seed"]) == 2
    for sampler in ("reverse", "single_shot"):
        assert res["samplers"][sampler]["invention_rate"]["k"] == sum(
            ps["samplers"][sampler]["invention_k"] for ps in res["per_seed"])
        assert res["samplers"][sampler]["solve"]["policy"]["k"] == sum(
            ps["samplers"][sampler]["solve_k"] for ps in res["per_seed"])
    assert res["arms"]["enumeration"]["solve"]["k"] == sum(
        ps["enumeration"]["solve_k"] for ps in res["per_seed"])
    assert res["samplers"]["reverse"]["invention_rate"]["n"] == 4  # 2 seeds x n=2
    # declared Holm family: base 4 tests when guarded arms are absent
    ch = res["comparisons_holm"]
    assert ch["method"] == "holm" and ch["alpha"] == 0.05
    assert set(ch["tests"]) == {
        "reverse:policy_vs_random", "reverse:policy_vs_fixed",
        "single_shot:policy_vs_random", "single_shot:policy_vs_fixed",
    }
    assert ch["m"] == 4
    for t in ch["tests"].values():
        assert set(t) == {"z", "p", "p_holm", "significant_05"}
        assert t["p_holm"] >= t["p"]
    assert "policy_forward_s_mean" in res["timing"]
    assert "enumeration_s_mean" in res["timing"]
    assert "single training seed; CIs cover instance variance only" in res["caveats"]
    assert res["positive_control"]["certificates"] == {"exact": 4}


# --- 5: Holm known values ----------------------------------------------------


def test_holm_known_values(mod):
    pvals = [0.01, 0.04, 0.03, 0.005]
    adj = mod._holm(pvals)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # monotone in ascending-p order and capped at 1
    order = sorted(range(4), key=lambda i: pvals[i])
    ordered = [adj[i] for i in order]
    assert ordered == sorted(ordered)
    assert mod._holm([0.9, 0.95]) == [1.0, 1.0]
    assert mod._holm([]) == []


# --- 6: enumeration on certified toys ----------------------------------------


def test_enumeration_solves_certified_toys(mod):
    res = mod.evaluate_full(_GoldStub(), data="toys", n=2, K=3, k_refine=1,
                            T=6, eval_seeds=[900000, 901000], ckpt=None)
    e = res["arms"]["enumeration"]
    assert e["status"] == "ok"
    assert e["solve"]["rate"] == 1.0
    assert e["first_accept_is_gold"]["rate"] == 1.0
    assert e["solved_calls_mean"] >= 1.0
    assert e["refine_calls_mean"] == e["solved_calls_mean"]  # k_refine=1
    assert e["wall_clock_s"]["total"] > 0.0


# --- 7: guarded arms skip cleanly when P1/P2 absent --------------------------


def test_guarded_arms_skip_cleanly(mod, monkeypatch):
    class _NoAblate:
        def __init__(self, D=16, L=1, K=3):
            pass

    monkeypatch.setattr(mod, "StructurePolicy", _NoAblate)
    monkeypatch.delattr(mod._policy_mod, "predicted_pin", raising=False)
    res = mod.evaluate_full(
        _GoldStub(), data="toys", n=1, K=3, k_refine=1, T=6,
        eval_seeds=[900000], ckpt={"model_kwargs": {}, "model_state_dict": {}},
    )
    nc = res["arms"]["no_context"]
    assert nc["status"] == "skipped"
    assert nc["reason"] == "StructurePolicy lacks ablate_context (W3 not merged)"
    pv = res["arms"]["policy_value"]
    assert pv["status"] == "skipped"
    assert "predicted_pin" in pv["reason"]
    # skipped arms never enter the Holm family
    assert "reverse:policy_vs_no_context" not in res["comparisons_holm"]["tests"]
    assert "reverse:policy_value_vs_random" not in res["comparisons_holm"]["tests"]


# --- 8: n > SEED_STRIDE errors -----------------------------------------------


def test_n_exceeding_stride_errors(mod, tmp_path):
    assert mod.SEED_STRIDE == 1000
    r = subprocess.run(
        [sys.executable, SCRIPT, "--ckpt", str(tmp_path / "x.pt"),
         "--out", str(tmp_path / "o.json"), "--n", "1001"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 2
    assert "SEED_STRIDE" in r.stderr


# --- pinned contracts --------------------------------------------------------


def test_reference_solver_pinned(mod):
    assert mod.REFERENCE_SOLVER == {"name": "refine", "k_refine": 4, "polish_steps": 4000}


def test_seed_space_constants(mod):
    assert (mod.TEST_SEED_BASE, mod.SEED_STRIDE) == (900000, 1000)
    assert (mod.LEGACY_VAL_OFFSET, mod.LEGACY_VAL_SIZE) == (500000, 50)
