"""Every named real system is well-formed and has a real solution the strong
classical baseline finds within tolerance (external-validity suite).

Fast: scipy LM at modest budget. This guards the suite against a typo that would
make a system unsolvable (and therefore a misleading 0% row in the eval)."""
import sympy as sp

from marc.data.real_systems import real_systems
from marc.eval.runner import Problem
from marc.eval.solver import ScipySolver

TOL = 1e-6


def _max_resid(graph, x):
    vals = {v.id: xi for v, xi in zip(graph.variables, x)}
    return max(abs(float(sp.sympify(f.expression).subs(vals))) for f in graph.factors)


def test_suite_nonempty_and_shaped():
    systems = real_systems()
    assert len(systems) >= 6
    names = [s.name for s in systems]
    assert len(names) == len(set(names))  # unique
    for s in systems:
        assert len(s.graph.variables) >= 2
        assert len(s.graph.factors) >= 2
        assert s.domain and s.note
        # every edge references a real variable and factor
        vids = {v.id for v in s.graph.variables}
        fids = {f.id for f in s.graph.factors}
        for e in s.graph.edges:
            assert e.variable_id in vids and e.factor_id in fids


def test_every_system_is_real_solvable():
    for s in real_systems():
        g = s.graph
        solver = ScipySolver(seed=0, init_scale=s.init_scale)
        cands = solver.sample(Problem(id=s.name, graph=g, solution=[0.0] * len(g.variables)), 30)
        best = min((_max_resid(g, c) for c in cands if c is not None), default=float("inf"))
        assert best < TOL, f"{s.name}: no real solution found (min residual {best:.2e})"
