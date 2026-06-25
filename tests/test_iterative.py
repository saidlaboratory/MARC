"""Tests for marc/refine/iterative.py — energy-gradient (Langevin) refinement."""

import math

from marc.eval.problems import entrapment_suite, in_distribution
from marc.refine.iterative import build_energy_fns, refine


def test_energy_and_grad_match_residual():
    # E = 1/2 (x - 3)^2  ->  E(3)=0, grad(3)=0, E(0)=4.5, grad(0)=-3
    p = in_distribution(1)[0]
    e_fn, g_fn, n = build_energy_fns(p.graph)
    assert n == 2
    e_sol = e_fn(*p.solution)
    assert e_sol < 1e-9
    grad = g_fn(*p.solution)
    assert all(abs(g) < 1e-9 for g in grad)


def test_refine_solves_convex_linear():
    p = in_distribution(1)[0]
    trace = refine(p.graph, [0.0, 0.0], noise=True, seed=0)
    assert trace.converged
    assert trace.best_energy < 1e-12
    for got, want in zip(trace.x, p.solution):
        assert abs(got - want) < 1e-6


def test_noise_off_is_deterministic():
    p = in_distribution(1)[0]
    a = refine(p.graph, [0.3, -0.2], noise=False, seed=1)
    b = refine(p.graph, [0.3, -0.2], noise=False, seed=999)
    assert a.x == b.x  # seed irrelevant without noise


def test_gradient_descent_traps_but_noise_escapes():
    # On a trap problem from its spurious-basin start, plain GD stalls (E > tol)
    # while at least one noisy seed reaches the solution.
    p = entrapment_suite(5)[0]
    init = p.metadata["init"]
    off = refine(p.graph, init, noise=False, steps=1000, lr=0.02, sigma0=2.5, seed=0)
    assert off.best_energy > 1e-6  # trapped by construction

    escaped = any(
        refine(p.graph, init, noise=True, steps=1500, lr=0.02, sigma0=2.5, seed=s).converged
        for s in range(8)
    )
    assert escaped


def test_trace_is_serialisable_and_finite():
    p = in_distribution(1)[0]
    d = refine(p.graph, [0.0, 0.0], seed=0).to_dict()
    assert math.isfinite(d["best_energy"])
    assert d["n_steps"] > 0
