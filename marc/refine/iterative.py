"""Energy-gradient iterative refinement with optional injected noise.

This is the **[MVP]** solver from TECHNICAL_GUIDE §3.4 with the learned GNN
``g_theta`` replaced by the exact energy gradient from the CAS:

    x^{(k+1)} = x^{(k)} - lr * grad E(x^{(k)}) + sigma_k * xi,   xi ~ N(0, I)

* ``noise=False`` is the **deterministic constraint-relaxation baseline** named in
  §11 — plain gradient descent on the energy E = 1/2 sum_i r_i(x)^2. It stalls at
  locally-consistent-but-globally-wrong fixed points (grad E = 0 with E > 0).
* ``noise=True`` is the Langevin variant: injected, annealed noise lets the
  relaxation escape those fixed points (§3.1, §4). This is the load-bearing
  ingredient the noise-on/off ablation (RQ2) measures.

Davin's learned ``solve()`` drops into the exact same ``Solver`` contract
(``marc.eval.solver``); this module gives the harness a real, non-dummy solver to
produce P1 numbers before the checkpoint lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np
import sympy as sp

from marc.cas.checker import _residual_and_kind
from marc.graph.graph import FactorGraph


def build_energy_fns(
    graph: FactorGraph,
) -> tuple[Callable[..., float], Callable[..., list], int]:
    """Compile (energy, gradient, n_vars) callables for a graph via sympy lambdify.

    Energy is E = 1/2 sum_i r_i(x)^2 over the per-factor residuals (§6). Equality
    factors contribute g; inequality factors contribute the hinge max(0, g) so a
    satisfied inequality adds nothing. The compiled functions take the variable
    values positionally, in ``graph.variables`` order.
    """
    symbols = [sp.Symbol(v.id) for v in graph.variables]
    terms = []
    for f in graph.factors:
        g, kind = _residual_and_kind(sp.sympify(f.expression))
        terms.append(g if kind == "eq" else sp.Max(0, g))
    energy = sp.Rational(1, 2) * sum(t ** 2 for t in terms) if terms else sp.Integer(0)
    grad = [sp.diff(energy, s) for s in symbols]

    e_fn = sp.lambdify(symbols, energy, "numpy")
    g_fn = sp.lambdify(symbols, grad, "numpy")
    return e_fn, g_fn, len(symbols)


@dataclass
class RefineTrace:
    """Outcome of one refinement run."""

    x: List[float]                 # best (lowest-energy) iterate seen
    final_x: List[float]           # last iterate
    best_energy: float             # energy at ``x``
    final_energy: float            # energy at ``final_x``
    energies: List[float]          # energy at every step (length = steps + 1)
    noise: bool = False
    converged: bool = False        # best_energy <= tol

    def to_dict(self) -> dict:
        return {
            "x": [round(v, 8) for v in self.x],
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "noise": self.noise,
            "converged": self.converged,
            "n_steps": len(self.energies) - 1,
        }


def refine(
    graph: FactorGraph,
    x0: Sequence[float],
    *,
    steps: int = 300,
    lr: float = 0.05,
    sigma0: float = 0.5,
    noise: bool = True,
    anneal: bool = True,
    polish_steps: int = 400,
    polish_lr: float = 0.2,
    max_step_norm: float = 5.0,
    tol: float = 1e-6,
    seed: int | None = None,
) -> RefineTrace:
    """Run gradient (Langevin) refinement from ``x0`` and return the trajectory.

    The step is clipped to ``max_step_norm`` for stability on stiff (e.g. cubic)
    residuals. ``sigma0`` is the initial noise scale, linearly annealed to 0 over
    ``steps`` when ``anneal`` is set. After exploration, ``polish_steps`` of
    noise-free gradient descent run from the best iterate so far — the analogue of a
    diffusion schedule's noise→0 tail — tightening convergence to the precision the
    symbolic checker needs (and *not* changing which basin the iterate is in, so the
    entrapment classification is unaffected). Returns the best iterate seen, not just
    the last — a downstream checker would accept the moment energy hits zero.
    """
    e_fn, g_fn, n = build_energy_fns(graph)
    rng = np.random.default_rng(seed)

    x = np.asarray(x0, dtype=float).copy()
    if x.shape != (n,):
        raise ValueError(f"x0 has length {x.size} but graph has {n} variables")

    def energy_at(v: np.ndarray) -> float:
        return float(e_fn(*v))

    e = energy_at(x)
    energies = [e]
    best_x, best_e = x.copy(), e

    for k in range(steps):
        grad = np.asarray(g_fn(*x), dtype=float).reshape(n)
        if not np.all(np.isfinite(grad)):
            break
        step = -lr * grad
        if noise:
            sigma = sigma0 * (1.0 - k / steps) if anneal else sigma0
            step = step + sigma * rng.standard_normal(n)
        # clip step magnitude to keep stiff gradients from blowing up
        norm = float(np.linalg.norm(step))
        if norm > max_step_norm:
            step *= max_step_norm / norm
        x = x + step
        e = energy_at(x)
        if not np.isfinite(e):
            break
        energies.append(e)
        if e < best_e:
            best_x, best_e = x.copy(), e

    # deterministic polish from the best iterate (noise→0 tail) for tight convergence.
    # The best iterate sits at a local min (grad ~ 0), so polishing cannot move it to
    # a different basin — it only tightens convergence within the current basin.
    x = best_x.copy()
    for _ in range(polish_steps):
        grad = np.asarray(g_fn(*x), dtype=float).reshape(n)
        if not np.all(np.isfinite(grad)):
            break
        step = -polish_lr * grad
        norm = float(np.linalg.norm(step))
        if norm > max_step_norm:
            step *= max_step_norm / norm
        x = x + step
        e = energy_at(x)
        if not np.isfinite(e):
            break
        energies.append(e)
        if e < best_e:
            best_x, best_e = x.copy(), e

    return RefineTrace(
        x=best_x.tolist(),
        final_x=x.tolist(),
        best_energy=best_e,
        final_energy=energies[-1],
        energies=energies,
        noise=noise,
        converged=best_e <= tol,
    )
