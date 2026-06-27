"""The three headline paper suites (TECHNICAL_GUIDE §11).

Each ``run_*`` function takes a solver honouring the :class:`marc.eval.solver.Solver`
contract (``refine`` for the baseline today, ``learned`` once Quang's Stage-B
checkpoint lands) and returns a JSON-serialisable dict. No plotting here — the
dicts are the contract between ``run_main_eval.py`` and ``plot_results.py``.

* :func:`run_generalization_gap` — in-distribution (2-var) vs. held-out-structure
  (3-var) solve rates and their gap (H1, derive-not-recall).
* :func:`run_perturbation`       — sweep the constant-perturbation magnitude and
  measure how much each split's solve rate drops when the solver must re-derive on
  shifted constants. A deriving solver stays flat; a memoriser falls off.
* :func:`run_length_extrapolation` — solve rate as a function of system length
  (#variables), trained at length 2 and probed out to longer chains.
"""

from __future__ import annotations

from typing import Any, List

from marc.eval.problems import held_out_structure, in_distribution, linear_system
from marc.eval.runner import DummySolver, run_eval, run_split_eval


def _solver_name(solver: Any) -> str:
    return getattr(solver, "name", type(solver).__name__)


def run_generalization_gap(
    solver: Any | None = None,
    *,
    n_id: int = 25,
    n_ho: int = 25,
    k: int = 4,
    perturb_delta: float = 0.1,
) -> dict:
    """In-distribution vs. held-out-structure split metrics + the generalization gap."""
    solver = solver or DummySolver()
    id_problems = in_distribution(n=n_id)
    ho_problems = held_out_structure(n=n_ho)
    metrics = run_split_eval(
        id_problems,
        ho_problems,
        solver=solver,
        n_samples=k,
        perturb_delta=perturb_delta,
        solver_name=_solver_name(solver),
    )
    metrics["suite"] = "generalization_gap"
    metrics["k"] = k
    return metrics


def run_perturbation(
    solver: Any | None = None,
    *,
    deltas: List[float] | None = None,
    n_id: int = 25,
    n_ho: int = 25,
    k: int = 4,
) -> dict:
    """Sweep constant-perturbation magnitude; record the solve-rate drop per split.

    For each ``delta`` the runner shifts every factor constant and the solver must
    re-derive on the shifted problem. ``robustness`` is the solve-rate drop
    (baseline − perturbed): ~0 for a deriving solver, large for a memoriser.
    """
    solver = solver or DummySolver()
    deltas = deltas or [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    id_problems = in_distribution(n=n_id)
    ho_problems = held_out_structure(n=n_ho)

    rows: List[dict] = []
    for delta in deltas:
        metrics = run_split_eval(
            id_problems,
            ho_problems,
            solver=solver,
            n_samples=k,
            perturb_delta=delta,
            solver_name=_solver_name(solver),
        )
        idm = metrics["splits"]["in_distribution"]
        hom = metrics["splits"]["held_out_structure"]
        rows.append(
            {
                "delta": delta,
                "in_distribution": {
                    "solve_rate": idm["solve_rate"],
                    "robustness": idm["perturbation_robustness"],
                    "perturbed_solve_rate": idm["solve_rate"] - idm["perturbation_robustness"],
                },
                "held_out_structure": {
                    "solve_rate": hom["solve_rate"],
                    "robustness": hom["perturbation_robustness"],
                    "perturbed_solve_rate": hom["solve_rate"] - hom["perturbation_robustness"],
                },
            }
        )

    return {
        "suite": "perturbation",
        "solver": _solver_name(solver),
        "k": k,
        "deltas": deltas,
        "sweep": rows,
    }


def run_length_extrapolation(
    solver: Any | None = None,
    *,
    lengths: List[int] | None = None,
    n: int = 25,
    k: int = 4,
    train_length: int = 2,
) -> dict:
    """Solve rate vs. system length (#variables); trained at ``train_length``."""
    solver = solver or DummySolver()
    # Default range where the energy-gradient *reference* solver is a clean oracle
    # (solves ~100% to checker precision). Beyond ~length 7 plain gradient descent
    # loses precision to conditioning (a solver artifact, not a generalization
    # signal), so any roll-off there would be a confound; override --lengths to probe
    # further with a tighter solver (e.g. the learned model once it lands).
    lengths = lengths or [2, 3, 4, 5, 6]

    rows: List[dict] = []
    for L in lengths:
        problems = linear_system(n_vars=L, n=n)
        metrics = run_eval(problems, solver=solver, n_samples=k)
        rows.append(
            {
                "length": L,
                "n_problems": metrics["n_problems"],
                "solve_rate": metrics["solve_rate"],
                "pass_at_k": metrics["pass_at_k"],
                "extrapolation": L > train_length,
            }
        )

    return {
        "suite": "length_extrapolation",
        "solver": _solver_name(solver),
        "k": k,
        "train_length": train_length,
        "lengths": lengths,
        "sweep": rows,
    }
