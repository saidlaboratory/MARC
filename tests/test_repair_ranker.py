import random

import torch

from conftest import load_script

from marc.graph.semantics import build_semantic_heterodata
from marc.model.repair_ranker import (
    CANDIDATE_FEATURE_DIM,
    CandidateOnlyRanker,
    GraphRepairRanker,
    batch_candidate_graphs,
    candidate_features,
)
from marc.structure.invention_data import make_dataset


def test_repair_ranker_scores_a_whole_menu():
    instances = make_dataset("aux_required", 2, 17, K=4)
    batch = batch_candidate_graphs(instances, build_semantic_heterodata)
    scores = GraphRepairRanker(D=24, L=2)(batch)
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()


def test_candidate_only_control_has_no_fixed_graph_features():
    a = make_dataset("aux_required", 1, 21, K=4)[0]
    b = make_dataset("aux_required", 1, 121, K=4)[0]
    fa = candidate_features(a, a.candidates[0])
    fb = candidate_features(b, b.candidates[0])
    assert fa.shape == fb.shape == (CANDIDATE_FEATURE_DIM,)
    model = CandidateOnlyRanker(D=16)
    assert model(torch.stack([fa, fb])).shape == (2,)


def test_paired_mcnemar_uses_discordant_pairs():
    module = load_script("run_repair_ranker")
    insts = make_dataset("aux_required", 3, 77, K=4)
    rows = [
        {"pack": type("P", (), {"inst": insts[0]})(),
         "full": insts[0].gold_idx, "random": (insts[0].gold_idx + 1) % 4},
        {"pack": type("P", (), {"inst": insts[1]})(),
         "full": insts[1].gold_idx, "random": (insts[1].gold_idx + 1) % 4},
        {"pack": type("P", (), {"inst": insts[2]})(),
         "full": (insts[2].gold_idx + 1) % 4, "random": insts[2].gold_idx},
    ]
    block = module._paired_mcnemar(rows, "full", "random")
    assert block["full_only_correct"] == 2
    assert block["baseline_only_correct"] == 1
    assert block["p_one_sided_exact"] == 0.5


def test_seed_hygiene_is_computed_not_asserted():
    # the provenance block must record the REAL per-source seed ranges (the
    # +100000*sidx stride) and COUNT id overlaps, never hardcode 0
    from types import SimpleNamespace

    module = load_script("run_repair_ranker")
    mk = lambda ids: [SimpleNamespace(inst=SimpleNamespace(id=i)) for i in ids]
    splits = {"train": mk(["a", "b"]), "validation": mk(["c"]), "test": mk(["b", "d"])}
    h = module.seed_hygiene(splits, ["aux_required", "nonlinear"], 100, 10, 5, 8)
    assert h["overlap_instances"] == 1  # "b" appears in train and test
    r = h["per_source_seed_ranges"]["nonlinear"]
    assert r["train"] == [100100, 100110]
    assert r["validation"] == [600100, 600105]
    assert r["test"] == [1000100, 1000108]

    disjoint = {"train": mk(["a"]), "validation": None, "test": mk(["b"])}
    assert module.seed_hygiene(disjoint, ["aux_required"], 0, 1, 1, 1)["overlap_instances"] == 0
