import random
import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

from marc.graph.semantics import build_semantic_heterodata
from marc.model.repair_ranker import (
    CANDIDATE_FEATURE_DIM,
    CandidateOnlyRanker,
    GraphRepairRanker,
    candidate_features,
)
from marc.structure.invention_data import make_dataset


def test_repair_ranker_scores_a_whole_menu():
    instances = make_dataset("aux_required", 2, 17, K=4)
    batch = Batch.from_data_list([
        build_semantic_heterodata(candidate.apply(inst.fixed_graph))
        for inst in instances
        for candidate in inst.candidates
    ])
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
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_repair_ranker.py"
    spec = importlib.util.spec_from_file_location("run_repair_ranker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
