"""3D (DMDGP) geometry auxiliary-construction repair (#123).

The 3D counterpart of :mod:`marc.structure.geo_repair`. Where the 2D module fixes
each point by two distance circles (a binary reflection across a *line*, branch =
sign of a triangle area / Heron), the native distance-geometry setting is 3D:
:func:`marc.data.geometry.make_pruned_chain_3d` builds chains whose points are
sphere-sphere-sphere intersections (a binary reflection across the anchor *plane*),
and the branch is the sign of a **tetrahedron's signed volume**, whose magnitude is
the Cayley-Menger determinant of four points' GIVEN squared distances.

Vocabulary (a fixed function of the givens, never the solution):

* ``gauge`` pins -- the z-mirror choice for P_0: z0 = +-6V/(c*d), where V is the
  volume of the tetrahedron (A1, A2, A3, P_0). This is the chain's global
  reflection gauge (the 3D analogue of the 2D y0 = +-2A/c pin).
* ``branch`` pins -- the reflection choice at each sphere-sphere-sphere step: the
  signed volume of the tetrahedron (P_{i-1}, A2, A3, P_i) equals +-V_i, with |V_i|
  a Cayley-Menger constant. The two signs are mutually exclusive branches.
* ``cos`` lifts -- the polarization/law-of-cosines dot relation
  (P_{i-1}-A2).(P_i-A2) = (a2_{i-1}+a2_i-l_i)/2: redundant, bilinear, always
  consistent (kept identical in spirit to the 2D module).

This module does NOT import from geo_repair.py (edited under #120); it is a
self-contained parallel so nothing blocks on that PR.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp
import torch

from marc.cas.checker import Checker
from marc.data.geometry import make_pruned_chain_3d
from marc.eval.solver import load_solver
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.structure.invention_data import REFERENCE_SOLVER

#: bump whenever identical seeds generate different instances/vocabularies
GEO_REPAIR3D_VERSION: int = 1

#: purpose-stream separation (same convention as the 2D module)
STREAM_SALT: int = 10_000_019

KINDS: Tuple[str, ...] = ("gauge", "branch", "cos")


@dataclass(frozen=True)
class Construction3D:
    """One auxiliary construction: a derived factor appended to the 3D graph."""

    name: str
    kind: str              # "gauge" | "branch" | "cos"
    position: int          # chain step the construction talks about (0 = gauge)
    sign: float            # +1/-1 for pins, 0 for lifts
    expression: str        # appended factor, rational constants only
    variables: Tuple[str, ...]
    const: float           # |derived constant| (for candidate-only features)

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


def cayley_menger_vol_sq(d01, d02, d03, d12, d13, d23) -> float:
    """Squared volume of a tetrahedron from its six SQUARED edge lengths, via the
    288 V^2 Cayley-Menger determinant. Returns V^2 (may be ~0 for near-degenerate
    tetrahedra; callers gate on a positive threshold)."""
    M = sp.Matrix([
        [0,   1,   1,   1,   1],
        [1,   0,   d01, d02, d03],
        [1,   d01, 0,   d12, d13],
        [1,   d02, d12, 0,   d23],
        [1,   d03, d13, d23, 0],
    ])
    cm = float(M.det())
    return cm / 288.0


def _signed_volume_expr(prev: int, cur: int, c: float, d: float) -> sp.Expr:
    """6*V(P_prev, A2, A3, P_cur) as a sympy polynomial in the point coordinates,
    with A2=(c,0,0), A3=(0,d,0). = det[A2-P_prev; A3-P_prev; P_cur-P_prev]."""
    xp, yp, zp = sp.symbols(f"x{prev} y{prev} z{prev}")
    xc, yc, zc = sp.symbols(f"x{cur} y{cur} z{cur}")
    r1 = sp.Matrix([c - xp, -yp, -zp])
    r2 = sp.Matrix([-xp, d - yp, -zp])
    r3 = sp.Matrix([xc - xp, yc - yp, zc - zp])
    return sp.Matrix.hstack(r1, r2, r3).det()


def construction_vocabulary_3d(k: int, givens: Dict) -> List[Construction3D]:
    """Fixed vocabulary for a k-point 3D pruned chain, derived from GIVENS only."""
    out: List[Construction3D] = []
    c, d = givens["c"], givens["d"]
    a2, a3 = givens["anchor2_sqs"], givens["anchor3_sqs"]
    links = givens["link_sqs"]
    a2a3 = c * c + d * d  # squared A2-A3 distance

    # gauge: tetra (A1=origin, A2, A3, P_0); signed vol = c*d*z0/6  ->  z0 = +-6V/(c*d)
    vsq0 = cayley_menger_vol_sq(c * c, d * d, givens["origin_sq"], a2a3, a2[0], a3[0])
    if vsq0 > 1e-9:
        z0_mag = 6.0 * math.sqrt(vsq0) / (c * d)
        for sgn in (1.0, -1.0):
            out.append(Construction3D(
                f"gauge_z0_{'p' if sgn > 0 else 'm'}", "gauge", 0, sgn,
                f"z0 - ({sgn * z0_mag})", ("z0",), abs(z0_mag),
            ))

    for i in range(1, k):
        varnames = tuple(f"{ax}{j}" for j in (i - 1, i) for ax in "xyz")
        # branch: signed volume of tetra (P_{i-1}, A2, A3, P_i) = +- 6*sqrt(V^2)
        vsq = cayley_menger_vol_sq(a2[i - 1], a3[i - 1], links[i - 1], a2a3, a2[i], a3[i])
        if vsq > 1e-9:
            six_v = _signed_volume_expr(i - 1, i, c, d)
            mag = 6.0 * math.sqrt(vsq)
            for sgn in (1.0, -1.0):
                expr = sp.expand(six_v - sgn * mag)
                out.append(Construction3D(
                    f"branch{i}_{'p' if sgn > 0 else 'm'}", "branch", i, sgn,
                    str(expr), varnames, mag,
                ))
        # cos lift: (P_{i-1}-A2).(P_i-A2) = (a2_{i-1}+a2_i-link)/2
        dot = f"(x{i-1} - ({c}))*(x{i} - ({c})) + y{i-1}*y{i} + z{i-1}*z{i}"
        rhs = (a2[i - 1] + a2[i] - links[i - 1]) / 2.0
        out.append(Construction3D(
            f"cos{i}", "cos", i, 0.0, f"({dot}) - ({rhs})", varnames, abs(rhs),
        ))
    return out


def solve_graph_3d(graph: FactorGraph, *, seed: int, k_restarts: Optional[int] = None) -> bool:
    """One reference-protocol attempt: REFERENCE_SOLVER multistart + exact checker."""
    solver = load_solver(REFERENCE_SOLVER["name"], seed=seed)
    prob = SimpleNamespace(id="geo_repair3d", graph=graph,
                           solution=[0.0] * len(graph.variables), metadata={})
    k = REFERENCE_SOLVER["k_refine"] if k_restarts is None else k_restarts
    cands = [c for c in solver.sample(prob, k) if c is not None]
    return Checker().first_accepted(graph, cands) is not None


@dataclass
class GeoRepair3DInstance:
    """One reference-solver FAILURE plus its labeled 3D construction menu."""

    id: str
    seed: int
    k: int
    graph: FactorGraph
    givens: Dict
    constructions: List[Construction3D]
    worked: List[bool]
    solution: List[float] = field(default_factory=list)  # audit only; never featurized


def givens_hash_3d(givens: Dict) -> str:
    """Content hash of an instance's given data (cross-split duplicate check)."""
    import hashlib
    key = (givens["c"], givens["d"], givens["origin_sq"],
           tuple(givens["anchor2_sqs"]), tuple(givens["anchor3_sqs"]),
           tuple(givens["link_sqs"]), tuple(sorted(givens["extra"])))
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


def label_instance_3d(graph: FactorGraph, constructions: Sequence[Construction3D],
                      *, solve_seed: int) -> List[bool]:
    """Measured labels under common random numbers (one restart stream for all)."""
    return [solve_graph_3d(c.apply(graph), seed=solve_seed) for c in constructions]


def make_dataset_3d(n_per_k: int, seed: int, ks: Sequence[int] = (6, 8),
                    n_extra: Optional[int] = None) -> List[GeoRepair3DInstance]:
    """Generate 3D chains, keep the HARD two-stream reference-solver failures, label
    their menus. Two-stream failure selection (both streams must fail) is mandatory:
    single-stream selection keeps one-bad-draw instances that solve on any fresh
    stream and inflates every downstream arm -- the #120 lesson applies in 3D too."""
    out: List[GeoRepair3DInstance] = []
    for k in ks:
        for t in range(n_per_k):
            inst_seed = seed + 1000 * k + t
            graph, sol, givens = make_pruned_chain_3d(k, random.Random(inst_seed),
                                                      n_extra=n_extra)
            if (solve_graph_3d(graph, seed=inst_seed)
                    or solve_graph_3d(graph, seed=inst_seed + STREAM_SALT)):
                continue  # solved on either stream -> not a hard failure
            vocab = construction_vocabulary_3d(k, givens)
            worked = label_instance_3d(graph, vocab,
                                       solve_seed=inst_seed + 2 * STREAM_SALT)
            out.append(GeoRepair3DInstance(
                id=f"pruned_chain3d_k{k}_s{inst_seed}",
                seed=inst_seed, k=k, graph=graph, givens=givens,
                constructions=vocab, worked=worked, solution=sol,
            ))
    return out


CONSTRUCTION3D_FEATURE_DIM = len(KINDS) + 3


def construction_features_3d(cons: Construction3D, k: int) -> torch.Tensor:
    v = [1.0 if cons.kind == kd else 0.0 for kd in KINDS]
    v += [cons.position / max(k - 1, 1), cons.sign, math.log1p(cons.const)]
    return torch.tensor(v, dtype=torch.float32)
