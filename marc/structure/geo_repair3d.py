"""3D DMDGP construction-repair vocabulary — the native-3D counterpart of
:mod:`marc.structure.geo_repair`.

The pruned 3D chain (:func:`marc.data.geometry.make_pruned_chain_3d`) fixes each
point by three squared distances, so the reference solver faces the real DMDGP
discrete branch: three spheres meet in two points reflected across the plane of
their centers. The repair vocabulary is the same shape as 2D, lifted a
dimension and derived from the GIVENS alone:

* ``branch`` pins — the reflection at step i is signed by the volume of the
  tetrahedron (O, A, P_{i-1}, P_i). That signed volume is
  ``c*(y_{i-1} z_i - z_{i-1} y_i)`` (a determinant of the coordinates), and its
  magnitude 6V is a Cayley-Menger determinant of the six *given* squared
  distances — so ``= +-6V`` is derivable data and the sign is the branch.
* ``gauge`` pins — the same for P_0's global reflection across the z=0 plane of
  the three fixed references O, A, B: ``c**2 z_0 = +-6V_0``.
* ``cos`` lifts — the redundant, degree-lowering dot-product relation
  ``(P_i-O).(P_{i-1}-O) = (|P_i|^2 + |P_{i-1}|^2 - link_i)/2``.

Nothing is answer-planted: the vocabulary is a fixed function of the givens, an
instance enters the population only because the reference solver failed it under
two independent restart streams, and a construction's label is the measured
outcome of re-solving with it added (see :func:`make_dataset_3d`).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

from marc.data.geometry import make_pruned_chain_3d
from marc.structure.geo_repair import (
    GeoRepairInstance, STREAM_SALT, Construction, label_instance, solve_graph,
)

GEO_REPAIR3D_VERSION: int = 1


def _cayley_menger_6vol(d2: Dict) -> float:
    """6 x tetrahedron volume from the six pairwise squared distances of its four
    points, via the Cayley-Menger determinant (288 V^2 = det, so (6V)^2 = det/8).
    ``d2`` keys are frozensets {p,q} -> squared distance. Returns 0 on a
    degenerate/near-planar tetrahedron."""
    pts = sorted({p for key in d2 for p in key})
    n = len(pts)
    idx = {p: i for i, p in enumerate(pts)}
    m = [[0.0] * (n + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        m[0][j] = m[j][0] = 1.0
    for key, val in d2.items():
        a, b = (idx[p] + 1 for p in key)
        m[a][b] = m[b][a] = val
    det = _det(m)
    return math.sqrt(det / 8.0) if det > 1e-9 else 0.0


def _det(m: List[List[float]]) -> float:
    """Plain Laplace/elimination determinant (5x5 here — no numpy dependency)."""
    m = [row[:] for row in m]
    n = len(m)
    det = 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return 0.0
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
            det = -det
        det *= m[col][col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            if f:
                for cc in range(col, n):
                    m[r][cc] -= f * m[col][cc]
    return det


def construction_vocabulary_3d(k: int, givens: Dict) -> List[Construction]:
    """The fixed 3D vocabulary for a k-point pruned chain, derived from GIVENS only."""
    c = givens["c"]
    o, aA = givens["origin_sqs"], givens["anchorA_sqs"]
    link, aB0 = givens["link_sqs"], givens["anchorB_sq0"]
    out: List[Construction] = []

    # gauge: tetrahedron (O, A=(c,0,0), B=(0,c,0), P_0); c^2 z0 = +-6V_0
    v0 = _cayley_menger_6vol({
        frozenset(("O", "A")): c * c, frozenset(("O", "B")): c * c,
        frozenset(("A", "B")): 2 * c * c, frozenset(("O", "P0")): o[0],
        frozenset(("A", "P0")): aA[0], frozenset(("B", "P0")): aB0})
    if v0 > 1e-9:
        for sgn in (1.0, -1.0):
            out.append(Construction(
                f"gauge_z0_{'p' if sgn > 0 else 'm'}", "gauge", 0, sgn,
                f"({c * c})*z0 - ({sgn * v0})", ("z0",)))

    for i in range(1, k):
        # branch: tetrahedron (O, A, P_{i-1}, P_i); c*(y_{i-1} z_i - z_{i-1} y_i) = +-6V_i
        v = _cayley_menger_6vol({
            frozenset(("O", "A")): c * c,
            frozenset(("O", "Pp")): o[i - 1], frozenset(("O", "Pc")): o[i],
            frozenset(("A", "Pp")): aA[i - 1], frozenset(("A", "Pc")): aA[i],
            frozenset(("Pp", "Pc")): link[i - 1]})
        cross = f"({c})*(y{i-1}*z{i} - z{i-1}*y{i})"
        varnames = (f"y{i-1}", f"z{i-1}", f"y{i}", f"z{i}")
        if v > 1e-9:
            for sgn in (1.0, -1.0):
                out.append(Construction(
                    f"branch{i}_{'p' if sgn > 0 else 'm'}", "branch", i, sgn,
                    f"({cross}) - ({sgn * v})", varnames))
        # cos lift: (P_i - O).(P_{i-1} - O) = (|P_i|^2 + |P_{i-1}|^2 - link_i)/2
        rhs = (o[i] + o[i - 1] - link[i - 1]) / 2.0
        out.append(Construction(
            f"cos{i}", "cos", i, 0.0,
            f"x{i}*x{i-1} + y{i}*y{i-1} + z{i}*z{i-1} - ({rhs})",
            (f"x{i-1}", f"y{i-1}", f"z{i-1}", f"x{i}", f"y{i}", f"z{i}")))
    return out


def make_dataset_3d(n_per_k: int, seed: int, ks: Sequence[int] = (6, 8),
                    n_extra: Optional[int] = None, label_streams: int = 3,
                    label_restarts: Optional[int] = None) -> List[GeoRepairInstance]:
    """Generate 3D chains, keep the two-stream-hard reference-solver failures, label
    their menus — same discipline as :func:`marc.structure.geo_repair.make_dataset`
    (an instance is a failure only if the direct solve fails on two independent
    streams; purpose streams are STREAM_SALT-separated)."""
    out: List[GeoRepairInstance] = []
    for k in ks:
        for t in range(n_per_k):
            inst_seed = seed + 1000 * k + t
            graph, sol, givens = make_pruned_chain_3d(k, random.Random(inst_seed),
                                                      n_extra=n_extra)
            if (solve_graph(graph, seed=inst_seed)
                    or solve_graph(graph, seed=inst_seed + STREAM_SALT)):
                continue
            vocab = construction_vocabulary_3d(k, givens)
            worked = label_instance(graph, vocab, solve_seed=inst_seed + 2 * STREAM_SALT,
                                    streams=label_streams, k_restarts=label_restarts)
            out.append(GeoRepairInstance(
                id=f"pruned_chain3d_k{k}_s{inst_seed}", seed=inst_seed, k=k,
                graph=graph, givens=givens, constructions=vocab, worked=worked, solution=sol))
    return out
