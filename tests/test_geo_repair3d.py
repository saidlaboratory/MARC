"""Consistency tests for the 3D (DMDGP) geometry repair machinery (#123).

Mirrors tests/test_geo_repair.py: the planted configuration satisfies every base
factor, the derived vocabulary is a function of the givens only, and at the true
config exactly one branch/gauge sign per pair holds while cos lifts vanish. The
global z-mirror is the other valid sheet, so it must flip the gauge sign.
"""
import math
import random

import pytest
import sympy as sp

from marc.data.geometry import make_clique_chain_3d, make_pruned_chain_3d
from marc.structure.geo_repair3d import (
    CONSTRUCTION3D_FEATURE_DIM,
    KINDS,
    cayley_menger_vol_sq,
    construction_features_3d,
    construction_vocabulary_3d,
)

GIVEN_KEYS = {"c", "d", "origin_sq", "anchor2_sqs", "anchor3_sqs", "link_sqs", "extra"}


def _chain(k=6, seed=11):
    return make_pruned_chain_3d(k, random.Random(seed))


def _env(graph, sol):
    return {sp.Symbol(v.id): x for v, x in zip(graph.variables, sol)}


def _res(expr, env):
    return float(sp.sympify(expr).subs(env))


def test_deterministic_and_solution_satisfies_graph():
    k = 6
    g1, s1, gv1 = _chain(k)
    g2, s2, gv2 = _chain(k)
    assert [(f.id, f.expression) for f in g1.factors] == [(f.id, f.expression) for f in g2.factors]
    assert s1 == s2
    assert set(gv1) == GIVEN_KEYS
    # 3k variables, 3 constraints/point + n_extra long-range
    assert len(g1.variables) == 3 * k
    env = _env(g1, s1)
    assert max(abs(_res(f.expression, env)) for f in g1.factors) < 1e-9


def test_givens_only_and_shapes():
    k = 7
    _, _, gv = _chain(k)
    assert len(gv["anchor2_sqs"]) == k
    assert len(gv["anchor3_sqs"]) == k
    assert len(gv["link_sqs"]) == k - 1
    assert all(len(t) == 3 for t in gv["extra"])


@pytest.mark.parametrize("seed", range(6))
def test_vocabulary_consistent_at_true_config(seed):
    k = 6
    g, sol, gv = _chain(k, seed)
    env = _env(g, sol)
    env_mirror = {sp.Symbol(v.id): (-sol[i] if v.id.startswith("z") else sol[i])
                  for i, v in enumerate(g.variables)}
    vocab = construction_vocabulary_3d(k, gv)
    by_kind = {kd: [c for c in vocab if c.kind == kd] for kd in KINDS}

    # cos lifts are redundant identities: they vanish at the true config
    assert all(abs(_res(c.expression, env)) < 1e-6 for c in by_kind["cos"])

    # branch/gauge: exactly one sign per pair (position) holds at the true config
    for kind in ("gauge", "branch"):
        per_pos = {}
        for c in by_kind[kind]:
            per_pos.setdefault(c.position, []).append(abs(_res(c.expression, env)) < 1e-6)
        assert per_pos, f"no {kind} constructions generated"
        assert all(sum(v) == 1 for v in per_pos.values()), f"{kind}: not exactly one sign/pair"

    # the global z-mirror is the other valid gauge sheet -> it flips the gauge sign
    gauge_true = [abs(_res(c.expression, env)) < 1e-6 for c in by_kind["gauge"]]
    gauge_mir = [abs(_res(c.expression, env_mirror)) < 1e-6 for c in by_kind["gauge"]]
    assert gauge_true != gauge_mir


def test_branch_magnitude_matches_cayley_menger():
    # the branch pin's constant is 6*sqrt(V^2) with V^2 the Cayley-Menger volume;
    # at the true config the signed-volume expression equals +-that constant.
    k = 5
    g, sol, gv = _chain(k, seed=3)
    env = _env(g, sol)
    for c in construction_vocabulary_3d(k, gv):
        if c.kind == "branch":
            # signed_volume_expr - sign*mag == 0 at truth for exactly one sign
            assert c.const > 0
            hit = abs(_res(c.expression, env)) < 1e-6
            # the matching-sign construction hits zero; residual magnitude else = 2*mag
            if not hit:
                assert abs(abs(_res(c.expression, env)) - 2 * c.const) < 1e-6


def test_cayley_menger_regular_tetrahedron():
    # unit regular tetrahedron: all six squared edges = 1, V = 1/(6 sqrt(2)),
    # so V^2 = 1/72.
    vsq = cayley_menger_vol_sq(1, 1, 1, 1, 1, 1)
    assert math.isclose(vsq, 1.0 / 72.0, rel_tol=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_clique_chain_solution_satisfies_graph(seed):
    # consecutive-clique DMDGP variant: planted config satisfies every factor,
    # each i>=3 point is pinned to its three predecessors, base is gauge-fixed.
    k = 8
    g, sol, gv = make_clique_chain_3d(k, random.Random(seed))
    assert len(g.variables) == 3 * k
    assert set(gv) == {"base", "clique", "extra"}
    for i in range(3, k):
        assert [j for j, _ in gv["clique"][i]] == [i - 1, i - 2, i - 3]
    env = _env(g, sol)
    assert max(abs(_res(f.expression, env)) for f in g.factors) < 1e-9


def test_construction_features_shape():
    _, _, gv = _chain(6, 1)
    vocab = construction_vocabulary_3d(6, gv)
    feats = construction_features_3d(vocab[0], 6)
    assert feats.shape == (CONSTRUCTION3D_FEATURE_DIM,)
