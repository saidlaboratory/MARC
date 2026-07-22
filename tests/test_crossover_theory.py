"""Property + smoke tests for the factorization-law experiment (R9).

Fast: tiny trial counts. Verifies the algebra of the law (best-of-K identity,
log-linear slope recovery), that the two reachability estimators are consistent
(best-of-K >= single-start), and the qualitative dichotomy the law rests on —
the independent (separable) family's log-reachability slope is markedly steeper
than the coupled family's. The full parameter-free validation is produced by
``scripts/run_crossover_theory.py`` (too heavy for the unit suite)."""
import math

from conftest import load_script

rct = load_script("run_crossover_theory")


def test_best_of_k_identity():
    assert rct.best_of_k(0.0, 8) == 0.0
    assert rct.best_of_k(1.0, 8) == 1.0
    # 1-(1-q)^K, monotone increasing in both q and K
    assert rct.best_of_k(0.2, 8) > rct.best_of_k(0.2, 4) > rct.best_of_k(0.2, 1)
    assert math.isclose(rct.best_of_k(0.5, 3), 1 - 0.5 ** 3)


def test_loglin_recovers_geometric_slope():
    # synthetic q(n) = v^n should fit with slope log v and R^2 ~ 1
    v = 0.4
    ns = [1, 2, 3, 4]
    qs = [v ** n for n in ns]
    a, b, r2, used = rct._loglin_fit(ns, qs)
    assert math.isclose(b, math.log(v), abs_tol=1e-9)
    assert r2 > 0.999
    assert used == ns


def test_loglin_skips_zero_reachability():
    a, b, r2, used = rct._loglin_fit([1, 2, 3], [0.5, 0.25, 0.0])
    assert used == [1, 2]  # the q=0 point is dropped, not logged


def test_single_start_and_bestofk_consistent():
    # best-of-K reachability must dominate a single start (same conditions)
    k1, t1 = rct.single_start_q("indep", 2, 20, rct.INDEP_START, seed0=7)
    kK, tK = rct.bestofk_random("indep", 2, 20, 8, rct.INDEP_START, seed0=7)
    assert 0 <= k1 <= t1 == 20
    assert 0 <= kK <= tK == 20
    assert kK / tK >= k1 / t1 - 1e-9


def test_geometry_point_chain_generator():
    # the scalable coupled geometry family: 2k variables, checker accepts the exact
    # integer-coordinate solution, and reachability is measurable (>0) at small k.
    import random
    from marc.data.geometry import make_point_chain
    from marc.cas.checker import Checker
    chk = Checker()
    for k in (1, 2, 3):
        g, sol = make_point_chain(k, random.Random(k))
        assert len(g.variables) == 2 * k
        assert len(g.factors) == 2 * k          # one coupling + one anchor per point
        assert chk.verify(g, sol).accepted
    ok, t = rct.single_start_q_geometry(1, 12, seed0=3)
    assert 0 <= ok <= t == 12 and ok > 0        # a single triangle is reachably solvable


def test_factorization_dichotomy_qualitative():
    # the crux: independent basins factorize (steep negative log-slope) while the
    # coupled chain does not (near-flat). Tiny trials -> assert the ordering only.
    ni = rct.measure_family("indep", [1, 2, 3, 4], rct.INDEP_START, 40, 8, seed0=11)
    nc = rct.measure_family("coupled", [2, 3, 4, 6], rct.COUPLED_START, 40, 8, seed0=13)
    bi = rct._loglin_fit([r["n"] for r in ni], [r["q"] for r in ni])[1]
    bc = rct._loglin_fit([r["n"] for r in nc], [r["q"] for r in nc])[1]
    assert bi < bc, f"independent slope {bi} should be steeper (more negative) than coupled {bc}"
    assert bi < -0.1, "independent reachability should decay geometrically"
