"""Solver contract for the eval harness, plus adapters and a loader.

The runner is solver-agnostic: it only needs an object exposing
``sample(problem, k) -> list[list[float]]`` (k candidate assignments). This module
pins that contract (``Solver``), exposes the real energy-gradient refinement solver
(``GradientRefinementSolver``), wraps the learned diffusion ``solve()`` into the
contract (``LearnedSolver``), keeps a generic single-call adapter
(``FunctionSolver``), and gives ``load_solver()`` one place to pick a solver with a
graceful, *loud* fallback so a green P1 run is never produced by a placeholder.

Plugging in the learned solver (P1, Day 3+): ``marc.diffusion.solve.solve`` already
exists on ``main`` with signature
``solve(graph, denoiser, cas_engine, steps, N, guidance_weight, eta, T) -> Tensor``.
Once Quang's checkpoint lands, run with ``--solver learned`` and
``MARC_CKPT=/path/to/denoiser.pt``. The learned path additionally needs
``torch_geometric`` installed (the GNN denoiser builds a ``HeteroData``).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any, Callable, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import sympy as sp

from marc.cas.checker import _residual_and_kind
from marc.refine.iterative import build_residual_jac, refine


@runtime_checkable
class Solver(Protocol):
    """Anything that proposes k candidate assignments for a problem."""

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        ...


def _downsample(values: Sequence[float], max_points: int = 50) -> List[float]:
    """Even-stride downsample to <= max_points, always keeping first and last."""
    if len(values) <= max_points:
        return list(values)
    step = (len(values) - 1) / (max_points - 1)
    return [values[round(i * step)] for i in range(max_points)]


def _trace_info(trace: Any) -> dict:
    """Per-solve diagnostics from a RefineTrace (the data the runner used to discard)."""
    return {
        "n_steps": len(trace.energies) - 1,
        "best_energy": float(trace.best_energy),
        "final_energy": float(trace.final_energy),
        "converged": bool(trace.converged),
        "energies": _downsample([float(e) for e in trace.energies]),
    }


class FunctionSolver:
    """Adapt a plain ``solve()`` callable to the ``Solver`` contract.

    ``solve_fn`` may accept either a ``Problem`` or a ``FactorGraph`` and return a
    single assignment (list/sequence of floats). k candidates come from calling it
    k times — meaningful when the underlying solver is stochastic (diffusion
    sampling); deterministic solvers just repeat their answer, leaving pass@k equal
    to pass@1, which is the honest result.
    """

    def __init__(
        self,
        solve_fn: Callable[[Any], Sequence[float]],
        *,
        pass_graph: bool = False,
        name: str = "function",
    ) -> None:
        self._solve = solve_fn
        self._pass_graph = pass_graph
        self.name = name

    def _solve_one(self, problem: Any) -> List[float]:
        arg = problem.graph if self._pass_graph else problem
        return [float(v) for v in self._solve(arg)]

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        return [self._solve_one(problem) for _ in range(k)]


class GradientRefinementSolver:
    """Real solver: energy-gradient Langevin refinement (TECHNICAL_GUIDE §3.4).

    Each candidate is an independent random restart refined toward E = 0, so pass@k
    genuinely improves with k. ``noise`` toggles the Langevin term — the same knob
    the entrapment ablation sweeps; the baseline eval leaves it on.
    """

    def __init__(
        self,
        *,
        steps: int = 300,
        lr: float = 0.05,
        sigma0: float = 0.5,
        noise: bool = True,
        init_scale: float = 3.0,
        polish_steps: int = 400,
        polish_lr: float = 0.2,
        seed: int | None = 0,
        name: str = "refine",
    ) -> None:
        self.steps = steps
        self.lr = lr
        self.sigma0 = sigma0
        self.noise = noise
        self.init_scale = init_scale
        self.polish_steps = polish_steps
        self.polish_lr = polish_lr
        self.seed = seed
        self.name = name
        self._rng = __import__("numpy").random.default_rng(seed)

    def _init(self, problem: Any) -> List[float]:
        n = len(problem.graph.variables)
        scale = problem.metadata.get("init_scale", self.init_scale)
        return (self._rng.standard_normal(n) * scale).tolist()

    def _refine_once(self, problem: Any):
        return refine(
            problem.graph,
            self._init(problem),
            steps=self.steps,
            lr=self.lr,
            sigma0=self.sigma0,
            noise=self.noise,
            polish_steps=self.polish_steps,
            polish_lr=self.polish_lr,
            seed=int(self._rng.integers(0, 2 ** 31 - 1)),
        )

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        return [self._refine_once(problem).x for _ in range(k)]

    def sample_with_info(
        self, problem: Any, k: int
    ) -> Tuple[List[List[float]], List[dict]]:
        """Like ``sample`` (same RNG stream, identical candidates for the same seed)
        but also returns the per-candidate RefineTrace diagnostics."""
        traces = [self._refine_once(problem) for _ in range(k)]
        return [t.x for t in traces], [_trace_info(t) for t in traces]


class ScipySolver:
    """Classical baseline: Levenberg–Marquardt / trust-region via scipy least_squares.

    The "why didn't you compare against Newton/LM?" column. Uses the analytic
    residual + Jacobian compiled by :func:`build_residual_jac` and the same
    multi-start protocol as :class:`GradientRefinementSolver`: each of the k
    candidates is one solve from an independent Gaussian init at ``init_scale``
    (per-problem ``metadata["init_scale"]`` honoured), so pass@k genuinely improves
    with k. ``method="lm"`` on square/overdetermined systems (scipy's requirement
    for LM), ``"trf"`` otherwise.
    """

    def __init__(
        self,
        *,
        init_scale: float = 3.0,
        tol: float = 1e-6,
        seed: int | None = 0,
        name: str = "lm",
    ) -> None:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise RuntimeError(
                f"ScipySolver needs scipy (listed in requirements.txt): {exc}. "
                "pip install scipy, or use --solver refine."
            ) from exc
        self._least_squares = least_squares
        self.init_scale = init_scale
        self.tol = tol
        self.seed = seed
        self.name = name
        self._rng = np.random.default_rng(seed)

    def _init(self, problem: Any) -> np.ndarray:
        n = len(problem.graph.variables)
        scale = problem.metadata.get("init_scale", self.init_scale)
        return self._rng.standard_normal(n) * scale

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        return self.sample_with_info(problem, k)[0]

    def sample_with_info(
        self, problem: Any, k: int
    ) -> Tuple[List[List[float]], List[dict]]:
        """``sample`` plus per-candidate ``{n_steps (nfev), final_energy, converged}``."""
        r_fn, j_fn, n = build_residual_jac(problem.graph)
        m = len(problem.graph.factors)
        fun = lambda x: np.asarray(r_fn(*x), dtype=float).reshape(m)
        jac = lambda x: np.asarray(j_fn(*x), dtype=float).reshape(m, n)
        method = "lm" if m >= n else "trf"  # scipy's "lm" requires m >= n
        cands: List[List[float]] = []
        infos: List[dict] = []
        for _ in range(k):
            x0 = self._init(problem)
            try:
                res = self._least_squares(fun, x0, jac=jac, method=method)
            except ValueError:
                # e.g. non-finite residual at the init (sqrt of a negative on
                # non-polynomial factors): count this start as a failed candidate
                # rather than crash a whole eval run
                cands.append([float(v) for v in x0])
                infos.append({"n_steps": 0, "final_energy": float("inf"), "converged": False})
                continue
            energy = 0.5 * float(np.sum(res.fun ** 2))  # E = 1/2 sum r^2, as elsewhere
            cands.append([float(v) for v in res.x])
            infos.append(
                {
                    "n_steps": int(res.nfev),
                    "final_energy": energy,
                    "converged": bool(energy <= self.tol),
                }
            )
        return cands, infos


class ExactLinearSolver:
    """Exact classical baseline for linear systems (numpy solves these exactly).

    ``sympy.linear_eq_to_matrix`` extracts (A, b) from the equality residuals;
    ``numpy.linalg.lstsq`` solves — exact (to machine precision) on consistent
    linear systems, minimum-norm least-squares otherwise. Deterministic, so the k
    candidates are k copies of the one solution (pass@k == pass@1, the honest
    result — same convention as FunctionSolver). On NONLINEAR input sympy raises
    (``NonlinearError``, a ``ValueError`` subclass); per the Solver protocol this
    returns NO candidates (``[]``) rather than crashing, so nonlinear eval rows
    simply score 0 without special-casing. Inequality factors likewise yield no
    candidates — this is a linear-equality solver only.
    """

    def __init__(self, *, tol: float = 1e-6, name: str = "exact") -> None:
        self.tol = tol
        self.name = name

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        return self.sample_with_info(problem, k)[0]

    def sample_with_info(
        self, problem: Any, k: int
    ) -> Tuple[List[List[float]], List[dict]]:
        graph = problem.graph
        symbols = [sp.Symbol(v.id) for v in graph.variables]
        residuals = []
        for f in graph.factors:
            g, kind = _residual_and_kind(sp.sympify(f.expression))
            if kind != "eq":
                return [], []  # inequality: not a linear system of equations
            residuals.append(g)
        try:
            A, b = sp.linear_eq_to_matrix(residuals, symbols)
        except (ValueError, sp.PolynomialError):
            # sympy NonlinearError (ValueError subclass): system is not linear
            return [], []
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        energy = 0.5 * float(np.sum((A @ x - b) ** 2))
        cand = [float(v) for v in x]
        info = {"n_steps": 1, "final_energy": energy, "converged": bool(energy <= self.tol)}
        return [list(cand) for _ in range(k)], [dict(info) for _ in range(k)]


def _cas_engine_for(graph: Any):
    """Build a CASEngine from a FactorGraph by serialising it to the JSON it reads."""
    from marc.cas.engine import CASEngine
    from marc.graph.serialize import save_graph

    symbol_names = [v.id for v in graph.variables]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        save_graph(graph, path)
        return CASEngine(path, symbol_names)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class LearnedSolver:
    """Adapt the learned diffusion ``solve()`` (TECHNICAL_GUIDE §3, §5) to the contract.

    Wraps ``marc.diffusion.solve.solve(graph, denoiser, cas_engine, ...)``. Each
    candidate is one best-of-N DDIM rollout, so pass@k draws k independent rollouts.
    Requires ``torch`` + ``torch_geometric`` (the GNN builds a ``HeteroData``) and a
    trained checkpoint — without one the denoiser is randomly initialised, so the
    constructor warns loudly rather than silently reporting near-zero solve rates.

    ``solve()`` returns ``None`` when every one of its N rollouts diverges to a
    non-finite energy (e.g. a checkpoint evaluated far outside the factor shapes it
    trained on). ``sample()`` passes that through as a ``None`` candidate rather than
    crashing — callers (:mod:`marc.eval.runner`'s ``Checker.verify`, or
    ``scripts/demo_end_to_end.py``) should skip ``None`` entries.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        steps: int = 50,
        N: int = 16,
        guidance_weight: float = 2.5,
        eta: float = 0.0,
        model_kwargs: dict | None = None,
        name: str = "learned",
        polish: bool = True,
    ) -> None:
        try:
            import torch
            from marc.diffusion.solve import solve
            from marc.model.denoiser import GraphDenoiser
        except ImportError as exc:  # torch / torch_geometric / model missing
            raise RuntimeError(
                f"LearnedSolver needs torch + torch_geometric + the model stack: {exc}. "
                "Install torch_geometric, or use --solver refine for the baseline."
            ) from exc

        self._torch = torch
        self._solve = solve
        self.steps = steps
        self.N = N
        self.guidance_weight = guidance_weight
        self.eta = eta
        self.name = name
        self.polish = polish

        checkpoint = checkpoint or os.environ.get("MARC_CKPT")
        ckpt_dict = None
        if checkpoint:
            ckpt_dict = torch.load(checkpoint, map_location="cpu")

        # Prefer model_kwargs from checkpoint so the architecture always matches
        if ckpt_dict and isinstance(ckpt_dict, dict) and "model_kwargs" in ckpt_dict:
            resolved_kwargs = {**ckpt_dict["model_kwargs"], **(model_kwargs or {})}
        else:
            resolved_kwargs = model_kwargs or {}

        self.denoiser = GraphDenoiser(**resolved_kwargs)

        if ckpt_dict is not None:
            state = ckpt_dict.get("model", ckpt_dict.get("model_state_dict", ckpt_dict)) if isinstance(ckpt_dict, dict) else ckpt_dict
            # strict=False so checkpoints predating newer conditioning params (e.g.
            # the zero-init incident-constant projection) still load and behave as
            # they did when trained. strict=False alone still raises on SIZE
            # mismatches (conditioning widened fac_encoder/output_mlp inputs), so
            # drop those tensors — loudly, since the affected modules stay at init.
            own = self.denoiser.state_dict()
            drop = [k for k, v in state.items()
                    if k in own and own[k].shape != v.shape]
            if drop:
                warnings.warn(
                    f"checkpoint {checkpoint} predates the current GraphDenoiser "
                    f"architecture: {len(drop)} size-mismatched tensors left at "
                    f"init ({', '.join(drop)}). Expect degraded proposals; the "
                    "polish step absorbs some of it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                state = {k: v for k, v in state.items() if k not in drop}
            self.denoiser.load_state_dict(state, strict=False)
        elif not checkpoint:
            warnings.warn(
                "LearnedSolver running with an UNTRAINED denoiser (no checkpoint). "
                "Results are not meaningful — set MARC_CKPT to Quang's checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.denoiser.eval()

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        return self.sample_with_info(problem, k)[0]

    def sample_with_info(
        self, problem: Any, k: int
    ) -> Tuple[List[List[float]], List[dict]]:
        """``sample`` plus what's cheaply available per candidate: the polish
        refine()'s RefineTrace diagnostics (``source: "polish"``), or
        ``{"diverged": true}`` when every rollout diverged, or ``{}`` with
        polish disabled (the diffusion path keeps no per-step energies)."""
        cas_engine = _cas_engine_for(problem.graph)
        out: List[List[float]] = []
        infos: List[dict] = []
        with self._torch.no_grad():
            for _ in range(k):
                x = self._solve(
                    problem.graph,
                    self.denoiser,
                    cas_engine,
                    steps=self.steps,
                    N=self.N,
                    guidance_weight=self.guidance_weight,
                    eta=self.eta,
                )
                if x is None:
                    # every rollout this call diverged to a non-finite energy
                    out.append(None)
                    infos.append({"diverged": True})
                    continue
                cand = [float(v) for v in x.squeeze(-1).reshape(-1).tolist()]
                info: dict = {}
                if self.polish:
                    trace = self._polish(problem.graph, cand)
                    cand = trace.x
                    info = {**_trace_info(trace), "source": "polish"}
                out.append(cand)
                infos.append(info)
        return out, infos

    def _polish(self, graph: Any, x0: List[float]):
        """Diffusion proposes, energy-descent disposes: refine the (possibly coarse
        or diverged) diffusion candidate to strict tolerance. Reuses the same
        Langevin refinement as the baseline solver, seeded at the diffusion output.
        Returns the full RefineTrace; callers take ``.x`` for the candidate."""
        from marc.refine.iterative import refine

        # guard against non-finite diffusion outputs before handing to the polisher
        if not all(abs(v) < 1e6 for v in x0):
            x0 = [0.0 for _ in x0]
        return refine(graph, x0, noise=True, seed=0)


def load_solver(name: str | None = None, **kwargs: Any) -> Solver:
    """Return a solver by name, honouring env config with a graceful, loud fallback.

    Names: ``"refine"`` (default, real gradient solver), ``"dummy"`` (oracle-ish
    sanity solver), ``"learned"``/``"davin"`` (the diffusion ``solve()`` + a
    checkpoint), ``"lm"`` (scipy Levenberg–Marquardt classical baseline),
    ``"exact"`` (exact linear-system solver; no candidates on nonlinear input).
    ``None`` reads ``MARC_SOLVER`` (default ``"refine"``). The learned
    path raises a clear, actionable error when torch_geometric or the checkpoint is
    missing, so a green P1 run is never produced by a placeholder.
    """
    name = (name or os.environ.get("MARC_SOLVER") or "refine").lower()

    if name == "refine":
        return GradientRefinementSolver(**kwargs)
    if name == "dummy":
        from marc.eval.runner import DummySolver

        return DummySolver(**kwargs)
    if name in ("learned", "davin"):
        return LearnedSolver(**kwargs)
    if name == "lm":
        return ScipySolver(**kwargs)
    if name == "exact":
        return ExactLinearSolver(**kwargs)

    raise ValueError(f"unknown solver '{name}' (expected refine|dummy|learned|lm|exact)")
