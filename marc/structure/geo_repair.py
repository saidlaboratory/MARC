"""Geometry auxiliary-construction repair (v0.4).

The real-domain analogue of menu-based structural repair: DMDGP-style pruned
point chains (:func:`marc.data.geometry.make_pruned_chain`) that the reference
pipeline FAILS to solve as posed, plus a fixed, domain-generic vocabulary of
auxiliary constructions derived from the given distance data by classical
geometric identities:

* ``branch`` pins — the reflection choice at each circle-circle intersection
  step, as an equality on the cross product (P_{i-1}-C) x (P_i-C): its
  magnitude is 2*Area of the triangle whose three squared sides are all GIVEN
  (Heron / Cayley-Menger), so ``cross = +-v`` is derivable data, and the sign
  is the classic discrete branch. The two signs are mutually exclusive.
* ``gauge`` pins — the same choice for the first point (y0 = +-2A/c), which is
  the chain's global mirror gauge.
* ``cos`` lifts — the law-of-cosines dot-product relation
  (P_{i-1}-C).(P_i-C) = (a_{i-1}+a_i-l_i)/2: redundant, bilinear
  (degree-lowering), always consistent.

Nothing here is answer-planted: the vocabulary is a fixed function of the
GIVENS (never the solution), an instance enters the dataset only because the
reference solver measurably failed on it under TWO independent restart streams
(single-stream failure selects on noise: about half of one-stream failures
solve on any fresh stream), and a construction's label is the measured outcome
of re-solving with it added. Several constructions may repair one instance
(the global mirror alone guarantees sign pairs can both work); training and
evaluation treat the label set, not a single gold index.

Disclosed simplifications versus molecular DMDGP: instances are 2D planar
(K=2 discretization, not 3D), distances are exact rationals (not noisy
intervals), the constraint pattern is a common-anchor path plus sparse
long-range edges (not consecutive-clique), and ground-truth configurations
have integer coordinates, so the exact checker accepts essentially the planted
configuration and its global mirror while rejecting irrational spurious
branches for free. The recipe-only control's features include each
construction's derived constant — a function of the givens, mirroring v0.3's
candidate-only semantics — so it is recipe-plus-derived-magnitude blind, not
fully problem-blind.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from marc.cas.checker import Checker
from marc.data.geometry import make_pruned_chain
from marc.eval.solver import load_solver
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.structure.invention_data import REFERENCE_SOLVER

#: bump whenever identical seeds generate different instances/vocabularies.
#: v3: labels are majority votes over label_streams independent restart streams
#: (single-stream labels were noisy enough that the ranker could not learn).
GEO_REPAIR_VERSION: int = 3

#: stream separation for every solver purpose; larger than any split-base
#: spacing so purpose streams can never collide across splits
STREAM_SALT: int = 10_000_019

KINDS: Tuple[str, ...] = ("gauge", "branch", "cos")


@dataclass(frozen=True)
class Construction:
    """One auxiliary construction: a derived factor appended to the graph."""

    name: str          # e.g. "branch3_p"
    kind: str          # "gauge" | "branch" | "cos"
    position: int      # chain step the construction talks about (0 = gauge)
    sign: float        # +1/-1 for pins, 0 for lifts
    expression: str    # the appended factor, rational constants only
    variables: Tuple[str, ...]

    def apply(self, graph: FactorGraph) -> FactorGraph:
        vs = [VariableNode(v.id, v.value) for v in graph.variables]
        fs = [FactorNode(f.id, f.expression) for f in graph.factors]
        es = [Edge(e.variable_id, e.factor_id, e.coefficient) for e in graph.edges]
        fid = f"aux_{self.name}"
        if any(f.id == fid for f in fs):
            raise ValueError(f"graph already has factor {fid}")
        fs.append(FactorNode(fid, self.expression))
        es += [Edge(v, fid, 1.0) for v in self.variables]
        return FactorGraph(variables=vs, factors=fs, edges=es)


def _heron_sq(a2: float, b2: float, l2: float) -> float:
    """(4*Area)^2 — the squared cross-product magnitude — of a triangle from
    its three SQUARED side lengths (the 16*Area^2 Heron/Cayley-Menger form)."""
    return 4.0 * a2 * b2 - (a2 + b2 - l2) ** 2


def construction_vocabulary(k: int, givens: Dict) -> List[Construction]:
    """The fixed vocabulary for a k-point pruned chain, derived from GIVENS only."""
    out: List[Construction] = []
    c = givens["c"]
    s = _heron_sq(givens["origin_sq"], c * c, givens["anchor_sqs"][0])
    if s > 1e-9:
        v = math.sqrt(s) / (2.0 * c)
        for sgn in (1.0, -1.0):
            out.append(Construction(
                f"gauge_y0_{'p' if sgn > 0 else 'm'}", "gauge", 0, sgn,
                f"y0 - ({sgn * v})", ("y0",),
            ))
    for i in range(1, k):
        a_prev = givens["anchor_sqs"][i - 1]
        a_cur = givens["anchor_sqs"][i]
        link = givens["link_sqs"][i - 1]
        cross = f"(x{i-1} - ({c}))*y{i} - (x{i} - ({c}))*y{i-1}"
        varnames = (f"x{i-1}", f"y{i-1}", f"x{i}", f"y{i}")
        s = _heron_sq(a_prev, a_cur, link)
        if s > 1e-9:
            v = math.sqrt(s) / 2.0
            for sgn in (1.0, -1.0):
                out.append(Construction(
                    f"branch{i}_{'p' if sgn > 0 else 'm'}", "branch", i, sgn,
                    f"({cross}) - ({sgn * v})", varnames,
                ))
        dot = f"(x{i-1} - ({c}))*(x{i} - ({c})) + y{i-1}*y{i}"
        rhs = (a_prev + a_cur - link) / 2.0
        out.append(Construction(
            f"cos{i}", "cos", i, 0.0, f"({dot}) - ({rhs})", varnames,
        ))
    return out


def solve_graph(graph: FactorGraph, *, seed: int, k_restarts: Optional[int] = None) -> bool:
    """One reference-protocol attempt: REFERENCE_SOLVER multistart + exact checker."""
    solver = load_solver(REFERENCE_SOLVER["name"], seed=seed)
    prob = SimpleNamespace(id="geo_repair", graph=graph,
                           solution=[0.0] * len(graph.variables), metadata={})
    k = REFERENCE_SOLVER["k_refine"] if k_restarts is None else k_restarts
    cands = [c for c in solver.sample(prob, k) if c is not None]
    return Checker().first_accepted(graph, cands) is not None


@dataclass
class GeoRepairInstance:
    """One reference-solver FAILURE plus its labeled construction menu."""

    id: str
    seed: int
    k: int
    graph: FactorGraph
    givens: Dict
    constructions: List[Construction]
    worked: List[bool]                 # measured: construction + re-solve flips it
    solution: List[float] = field(default_factory=list)  # audit only; never featurized


def givens_hash(givens: Dict) -> str:
    """Content hash of an instance's given data — the real cross-split
    duplicate check (id strings embed seeds and can never collide)."""
    import hashlib
    key = (givens["c"], givens["origin_sq"], tuple(givens["anchor_sqs"]),
           tuple(givens["link_sqs"]), tuple(sorted(givens["extra"])))
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


def label_instance(graph: FactorGraph, constructions: Sequence[Construction],
                   *, solve_seed: int, streams: int = 1) -> List[bool]:
    """Measured labels under common random numbers: within a stream every
    construction is graded with the same restarts, so differences come from
    the construction alone. With ``streams`` > 1 a construction's label is the
    majority vote across independent streams — stream-stable repairs, which
    denoises the training target (a single stream flips ~half its verdicts on
    a fresh stream)."""
    votes = [
        [solve_graph(c.apply(graph), seed=solve_seed + 97 * j) for c in constructions]
        for j in range(streams)
    ]
    need = streams // 2 + 1
    return [sum(v[i] for v in votes) >= need for i in range(len(constructions))]


def make_dataset(n_per_k: int, seed: int, ks: Sequence[int] = (6, 8),
                 n_extra: Optional[int] = None, label_streams: int = 1,
                 cache_dir: Optional[str] = "results/p_geo_repair/cache") -> List[GeoRepairInstance]:
    """Generate chains, keep the HARD reference-solver failures, label their menus.

    An instance is a failure only if the direct solve fails under two
    independent restart streams (selection on a single stream keeps instances
    whose failure was one bad draw — about half of those solve on any fresh
    stream, which would inflate every downstream arm). Deterministic per
    (n_per_k, seed, ks, n_extra). Purpose streams are STREAM_SALT-separated:
    failure tests at +0/+1 salt, labels at +2, evaluation at +3 and up.
    """
    cache_path = None
    if cache_dir is not None:
        import pickle
        from pathlib import Path
        key = f"v{GEO_REPAIR_VERSION}_n{n_per_k}_s{seed}_k{'-'.join(map(str, ks))}_e{n_extra}_ls{label_streams}"
        cache_path = Path(cache_dir) / f"{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
    out: List[GeoRepairInstance] = []
    for k in ks:
        for t in range(n_per_k):
            inst_seed = seed + 1000 * k + t
            graph, sol, givens = make_pruned_chain(k, random.Random(inst_seed),
                                                   n_extra=n_extra)
            if (solve_graph(graph, seed=inst_seed)
                    or solve_graph(graph, seed=inst_seed + STREAM_SALT)):
                continue  # direct solve succeeded on either stream -> not hard
            vocab = construction_vocabulary(k, givens)
            worked = label_instance(graph, vocab,
                                    solve_seed=inst_seed + 2 * STREAM_SALT,
                                    streams=label_streams)
            out.append(GeoRepairInstance(
                id=f"pruned_chain_k{k}_s{inst_seed}",
                seed=inst_seed, k=k, graph=graph, givens=givens,
                constructions=vocab, worked=worked, solution=sol,
            ))
    if cache_path is not None:
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(out, fh)
    return out


#: candidate-only control features: the construction RECIPE alone (kind,
#: normalized position, sign, magnitude of its derived constant) — no problem
#: graph, mirroring the v0.3 candidate-only semantics.
CONSTRUCTION_FEATURE_DIM = len(KINDS) + 3


def construction_features(cons: Construction, k: int) -> torch.Tensor:
    v = [1.0 if cons.kind == kd else 0.0 for kd in KINDS]
    const = abs(float(cons.expression.rsplit("- (", 1)[1].rstrip(")")))
    v += [cons.position / max(k - 1, 1), cons.sign, math.log1p(const)]
    return torch.tensor(v, dtype=torch.float32)


class ConstructionOnlyRanker(torch.nn.Module):
    """Candidate-only control: scores a construction recipe with no problem graph."""

    def __init__(self, D: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(CONSTRUCTION_FEATURE_DIM, D), torch.nn.ReLU(),
            torch.nn.Linear(D, D), torch.nn.ReLU(),
            torch.nn.Linear(D, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)
