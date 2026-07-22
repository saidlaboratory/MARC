import inspect
import math
import random

import pytest
import sympy as sp
import torch

from conftest import load_script

import marc.structure.geo_repair as geo_repair
from marc.data.geometry import build_triangle_graph, make_pruned_chain
from marc.graph.semantics import build_semantic_heterodata
from marc.structure.geo_repair import (
    CONSTRUCTION_FEATURE_DIM,
    KINDS,
    construction_features,
    construction_vocabulary,
    label_instance,
)

GIVEN_KEYS = {"c", "origin_sq", "anchor_sqs", "link_sqs", "extra"}


def _chain(k=6, seed=11):
    return make_pruned_chain(k, random.Random(seed))


def _env(graph, sol):
    return {sp.Symbol(v.id): x for v, x in zip(graph.variables, sol)}


def _res(expr, env):
    return float(sp.sympify(expr).subs(env))


def _heron(a2, b2, l2):
    return 4.0 * a2 * b2 - (a2 + b2 - l2) ** 2


def _pin_const(gv, kind, pos):
    if kind == "gauge":
        return math.sqrt(_heron(gv["origin_sq"], gv["c"] ** 2, gv["anchor_sqs"][0])) / (2 * gv["c"])
    return math.sqrt(_heron(gv["anchor_sqs"][pos - 1], gv["anchor_sqs"][pos],
                            gv["link_sqs"][pos - 1])) / 2.0


def test_pruned_chain_deterministic_and_solution_satisfies_graph():
    k = 6
    g1, s1, gv1 = _chain(k)
    g2, s2, gv2 = _chain(k)
    assert [(f.id, f.expression) for f in g1.factors] == [(f.id, f.expression) for f in g2.factors]
    assert [(e.variable_id, e.factor_id, e.coefficient) for e in g1.edges] == \
           [(e.variable_id, e.factor_id, e.coefficient) for e in g2.edges]
    assert s1 == s2
    assert gv1 == gv2

    assert set(gv1) == GIVEN_KEYS
    assert len(gv1["anchor_sqs"]) == k
    assert len(gv1["link_sqs"]) == k - 1
    assert len(gv1["extra"]) == (k + 1) // 2  # default n_extra

    env = _env(g1, s1)
    for f in g1.factors:
        assert abs(_res(f.expression, env)) < 1e-9, f.id


def test_vocabulary_derives_from_givens_only():
    # no solution in the signature, and same givens -> same vocabulary
    assert list(inspect.signature(construction_vocabulary).parameters) == ["k", "givens"]
    k = 6
    _, _, gv = _chain(k)
    vocab = construction_vocabulary(k, gv)
    assert [c.expression for c in construction_vocabulary(k, gv)] == \
           [c.expression for c in vocab]
    # scale all squared distances by 4 and c by 2: still a valid instance shape,
    # but every derived constant moves -> the vocabulary tracked the givens
    pert = dict(gv, c=gv["c"] * 2.0, origin_sq=gv["origin_sq"] * 4.0,
                anchor_sqs=[v * 4.0 for v in gv["anchor_sqs"]],
                link_sqs=[v * 4.0 for v in gv["link_sqs"]])
    pvocab = construction_vocabulary(k, pert)
    assert [c.name for c in pvocab] == [c.name for c in vocab]
    for a, b in zip(vocab, pvocab):
        assert a.expression != b.expression


def test_pins_and_lifts_at_true_solution():
    k = 6
    graph, sol, gv = _chain(k)
    env = _env(graph, sol)
    pairs = {}
    n_cos = 0
    for c in construction_vocabulary(k, gv):
        if c.kind == "cos":
            n_cos += 1
            assert c.sign == 0.0
            assert abs(_res(c.expression, env)) < 1e-6
        else:
            pairs.setdefault((c.kind, c.position), []).append(c)
    assert n_cos == k - 1
    assert ("gauge", 0) in pairs
    assert any(kind == "branch" for kind, _ in pairs)
    for (kind, pos), pair in pairs.items():
        assert sorted(c.sign for c in pair) == [-1.0, 1.0]
        v = _pin_const(gv, kind, pos)
        r = sorted(abs(_res(c.expression, env)) for c in pair)
        assert r[0] < 1e-6            # the true sign pins to zero
        assert r[1] == pytest.approx(2 * v, rel=1e-6)  # the other sits at 2v


def test_apply_appends_one_factor_and_is_pure():
    graph, _, gv = _chain(6, seed=5)
    cons = construction_vocabulary(6, gv)[0]
    nf, ne, nv = len(graph.factors), len(graph.edges), len(graph.variables)
    g2 = cons.apply(graph)
    assert (len(graph.factors), len(graph.edges)) == (nf, ne)  # original untouched
    assert len(g2.variables) == nv
    assert len(g2.factors) == nf + 1
    new = g2.factors[-1]
    assert new.id == f"aux_{cons.name}"
    assert new.expression == cons.expression
    assert [(e.variable_id, e.factor_id) for e in g2.edges[ne:]] == \
           [(v, new.id) for v in cons.variables]
    assert g2.factors[0] is not graph.factors[0]  # copies, not shared nodes
    with pytest.raises(ValueError):
        cons.apply(g2)


def test_label_instance_one_bool_per_construction(monkeypatch):
    graph, _, gv = _chain(6, seed=3)
    vocab = construction_vocabulary(6, gv)
    calls = []

    def fake(g, *, seed, k_restarts=None):
        calls.append((g, seed))
        return len(calls) % 2 == 0

    monkeypatch.setattr(geo_repair, "solve_graph", fake)
    out = label_instance(graph, vocab, solve_seed=99)
    assert out == [i % 2 == 1 for i in range(len(vocab))]
    assert all(isinstance(b, bool) for b in out)
    assert {s for _, s in calls} == {99}  # common random numbers across the menu
    assert all(len(g.factors) == len(graph.factors) + 1 for g, _ in calls)


def test_label_instance_passes_label_restarts(monkeypatch):
    graph, _, gv = _chain(6, seed=3)
    vocab = construction_vocabulary(6, gv)
    seen = []

    def fake(g, *, seed, k_restarts=None):
        seen.append(k_restarts)
        return False

    monkeypatch.setattr(geo_repair, "solve_graph", fake)
    label_instance(graph, vocab, solve_seed=99, k_restarts=1)
    assert set(seen) == {1}


def test_make_dataset_parallel_matches_serial():
    kw = dict(ks=(6,), label_streams=3, cache_dir=None)
    serial = geo_repair.make_dataset(3, 424242, **kw)
    par = geo_repair.make_dataset(3, 424242, workers=2, **kw)
    assert [i.id for i in par] == [i.id for i in serial]
    assert [i.worked for i in par] == [i.worked for i in serial]
    assert [i.givens for i in par] == [i.givens for i in serial]
    assert [[f.expression for f in i.graph.factors] for i in par] == \
           [[f.expression for f in i.graph.factors] for i in serial]


def test_construction_features_dim_onehot_and_constants():
    k = 6
    _, _, gv = _chain(k)
    vocab = construction_vocabulary(k, gv)
    assert {c.kind for c in vocab} == set(KINDS)
    assert {c.sign for c in vocab} == {1.0, -1.0, 0.0}
    for c in vocab:
        f = construction_features(c, k)
        assert f.shape == (CONSTRUCTION_FEATURE_DIM,)
        assert f[:3].tolist() == [float(c.kind == kd) for kd in KINDS]
        assert f[3].item() == pytest.approx(c.position / (k - 1))
        assert f[4].item() == c.sign
        if c.kind == "cos":
            i = c.position
            const = abs(gv["anchor_sqs"][i - 1] + gv["anchor_sqs"][i]
                        - gv["link_sqs"][i - 1]) / 2.0
        else:
            const = _pin_const(gv, c.kind, c.position)
        assert f[5].item() == pytest.approx(math.log1p(const), rel=1e-5)


def _two_packs(module):
    def pack(n, off, labels):
        return module.Packed(
            inst=None,
            graphs=[build_semantic_heterodata(build_triangle_graph(2.0, 5.0, 3.0))
                    for _ in range(n)],
            feats=torch.arange(off, off + n, dtype=torch.float32).unsqueeze(1),
            labels=torch.tensor(labels))

    return [pack(3, 0, [0.0, 1.0, 0.0]), pack(5, 3, [0.0, 0.0, 0.0, 0.0, 1.0])]


def test_flat_batch_handles_variable_k_menus():
    module = load_script("run_geo_repair")
    gb, feats, labels, sizes = module._flat_batch(_two_packs(module), torch.device("cpu"))
    assert sizes == [3, 5]
    assert gb.num_graphs == 8
    assert feats[:, 0].tolist() == list(range(8))  # pack order preserved
    assert labels.tolist() == [0, 1, 0, 0, 0, 0, 0, 1]


class _Fixed(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, *_):
        return self.scores


def test_top1_hit_rates_slices_per_instance():
    module = load_script("run_geo_repair")
    packs = _two_packs(module)
    # full: argmax 1 in menu A (working), local argmax 4 in menu B (working)
    full = _Fixed(torch.tensor([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]))
    # control: argmax 0 in both menus (both non-working)
    control = _Fixed(torch.tensor([9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]))
    f, c = module.top1_hit_rates(full, control, packs, batch_size=8,
                                 device=torch.device("cpu"))
    assert f == 1.0
    assert c == 0.0
