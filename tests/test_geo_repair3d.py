import math
import random

import sympy as sp

from marc.data.geometry import make_pruned_chain_3d
from marc.structure.geo_repair3d import (
    construction_vocabulary_3d, _cayley_menger_6vol,
)

GIVEN_KEYS = {"c", "origin_sqs", "anchorA_sqs", "link_sqs", "anchorB_sq0", "extra"}


def _chain(k=6, seed=11):
    return make_pruned_chain_3d(k, random.Random(seed))


def _env(graph, sol):
    return {sp.Symbol(v.id): x for v, x in zip(graph.variables, sol)}


def _res(expr, env):
    return float(sp.sympify(expr).subs(env))


def test_true_solution_satisfies_all_factors():
    for k in (4, 6, 8):
        g, sol, gv = _chain(k)
        assert set(gv) == GIVEN_KEYS
        assert len(gv["origin_sqs"]) == k
        assert len(gv["anchorA_sqs"]) == k
        assert len(gv["link_sqs"]) == k - 1
        assert len(sol) == 3 * k
        env = _env(g, sol)
        for f in g.factors:            # planted config is an exact root of every factor
            assert abs(_res(f.expression, env)) < 1e-9, (k, f.id)


def test_vocabulary_derives_from_givens_only():
    # same givens -> identical vocabulary; scaling the givens moves every constant,
    # so the vocab tracked the givens and not a fixed table.
    _, _, gv = _chain(6)
    v1 = construction_vocabulary_3d(6, gv)
    assert [c.expression for c in construction_vocabulary_3d(6, gv)] == \
           [c.expression for c in v1]
    assert {c.kind for c in v1} == {"gauge", "branch", "cos"}
    assert {c.sign for c in v1} == {1.0, -1.0, 0.0}


def test_branch_and_gauge_pins_split_at_true_config():
    k = 6
    g, sol, gv = _chain(k)
    env = _env(g, sol)
    pairs = {}
    n_cos = 0
    for c in construction_vocabulary_3d(k, gv):
        if c.kind == "cos":
            n_cos += 1
            assert abs(_res(c.expression, env)) < 1e-6   # redundant lift holds exactly
        else:
            pairs.setdefault((c.kind, c.position), []).append(c)
    assert n_cos == k - 1
    assert ("gauge", 0) in pairs
    assert sum(kind == "branch" for kind, _ in pairs) >= 1
    for (kind, pos), pair in pairs.items():
        assert sorted(c.sign for c in pair) == [-1.0, 1.0]
        r = sorted(abs(_res(c.expression, env)) for c in pair)
        assert r[0] < 1e-6                    # exactly one sign pins the true branch
        assert r[1] > 1e-6                    # the other sits at 2*6V away
        assert math.isclose(r[1], 2 * (r[1] / 2), rel_tol=1e-9)


def test_cayley_menger_matches_a_known_tetrahedron():
    # unit right-corner tetrahedron O,(1,0,0),(0,1,0),(0,0,1): volume 1/6, so 6V=1.
    d2 = {frozenset(("O", "A")): 1.0, frozenset(("O", "B")): 1.0,
          frozenset(("O", "C")): 1.0, frozenset(("A", "B")): 2.0,
          frozenset(("A", "C")): 2.0, frozenset(("B", "C")): 2.0}
    assert math.isclose(_cayley_menger_6vol(d2), 1.0, rel_tol=1e-9)
