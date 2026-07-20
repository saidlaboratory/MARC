"""CoupledChainTemplate (marc/data/coupled.py) wraps make_chain unchanged."""

import random

from marc.cas.checker import Checker
from marc.data.coupled import CoupledChainTemplate, make_chain


def test_template_shape():
    tmpl = CoupledChainTemplate(n=4)
    assert tmpl.name == "CoupledChain4"
    graph, sol = tmpl.generate(seed=0)
    assert len(graph.variables) == 4
    assert len(graph.factors) == 2 * (4 - 1)
    assert set(sol) == {f"x{i}" for i in range(4)}


def test_checker_accepts_gold():
    checker = Checker()
    for seed in range(5):
        graph, sol = CoupledChainTemplate(n=4).generate(seed=seed)
        assert checker.verify(graph, [sol[v.id] for v in graph.variables]).accepted


def test_deterministic():
    assert CoupledChainTemplate(n=5).generate(seed=11) == CoupledChainTemplate(n=5).generate(seed=11)


def test_parity_with_make_chain():
    graph, xs = make_chain(4, random.Random(7))
    tgraph, sol = CoupledChainTemplate(n=4).generate(seed=7)
    assert tgraph == graph
    assert sol == {f"x{i}": xs[i] for i in range(4)}
