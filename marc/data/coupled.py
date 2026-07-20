"""Coupled chained-bilinear constraint family (P_coupled).

``make_chain`` moved verbatim from scripts/run_coupled_eval.py so training code
can import it; the script re-imports it from here. RNG draw order is identical,
so existing coupled-eval results reproduce exactly.

    x_i + x_{i+1} = s_i,   x_i * x_{i+1} = p_i    (i = 0 .. n-2)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

_INT = [-3, -2, -1, 1, 2, 3]


def make_chain(n, rng):
    xs = [rng.choice(_INT) for _ in range(n)]
    vs = [VariableNode(f"x{i}", value=0.0) for i in range(n)]
    fs, es = [], []
    for i in range(n - 1):
        s, p = xs[i] + xs[i + 1], xs[i] * xs[i + 1]
        fs.append(FactorNode(f"sum{i}", f"x{i}+x{i+1}-({s})"))
        fs.append(FactorNode(f"prod{i}", f"x{i}*x{i+1}-({p})"))
        es += [Edge(f"x{i}", f"sum{i}", 1), Edge(f"x{i+1}", f"sum{i}", 1),
               Edge(f"x{i}", f"prod{i}", 1), Edge(f"x{i+1}", f"prod{i}", 1)]
    return FactorGraph(vs, fs, es), [float(v) for v in xs]


@dataclass
class CoupledChainTemplate:
    """Chained bilinear system as a generator template: n vars, 2(n-1) factors."""

    n: int = 4
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"CoupledChain{self.n}"

    def generate(self, seed: int = None) -> Tuple[FactorGraph, Dict[str, float]]:
        graph, xs = make_chain(self.n, random.Random(seed))
        return graph, {f"x{i}": xs[i] for i in range(self.n)}
