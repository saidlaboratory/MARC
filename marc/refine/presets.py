"""Shared refine() hyperparameter presets (issue #104).

The geometry-tuned polish recipe was copy-pasted across four scripts
(run_geometry_eval, run_pointchain_eval, demo_end_to_end, run_crossover_theory),
so a retune in one place could silently give arms different budgets. This module
is the single source of truth; scripts import from here and must not redefine
the numbers.

Why these values: the geometry domain's energy is a nonconvex quartic
(squared-distance factors are quadratic in the unknowns), so the default
``refine()`` hyperparameters — tuned against convex linear systems — solve ~0%
of geometry instances. Noise off, a smaller learning rate, and a much longer
polish are needed to reach the checker's exact-rational tolerance
(results/p4_scale/roadmap.md).
"""

from __future__ import annotations

#: full solver-level recipe (GradientRefinementSolver / load_solver("refine")).
GEOMETRY_REFINE_KWARGS = dict(
    steps=1200, lr=0.008, sigma0=0.0, noise=False,
    polish_steps=6000, polish_lr=0.02, init_scale=3.0,
)

#: the same deterministic descent for direct ``refine()`` calls, which do not
#: take the solver-level ``init_scale`` knob (callers draw their own starts).
GEOMETRY_POLISH_KWARGS = {k: v for k, v in GEOMETRY_REFINE_KWARGS.items()
                          if k != "init_scale"}

#: standard deviation for Gaussian starts, matched to ``init_scale``.
GEOMETRY_INIT_SD = GEOMETRY_REFINE_KWARGS["init_scale"]
