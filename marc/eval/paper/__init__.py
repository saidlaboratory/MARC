"""Paper-result eval suites (P2): generalization gap, perturbation, length extrapolation.

These wrap the §11 eval harness (:mod:`marc.eval.runner`) into the three headline
suites the paper reports, each returning a JSON-serialisable dict that
``scripts/run_main_eval.py`` writes to ``results/p2_main/`` and
``scripts/plot_results.py`` renders into ``paper/figures/``.
"""

from marc.eval.paper.suites import (
    run_generalization_gap,
    run_length_extrapolation,
    run_perturbation,
)

__all__ = [
    "run_generalization_gap",
    "run_length_extrapolation",
    "run_perturbation",
]
