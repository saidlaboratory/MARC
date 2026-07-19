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
from typing import Any, Callable, List, Protocol, Sequence, runtime_checkable

from marc.refine.iterative import refine


@runtime_checkable
class Solver(Protocol):
    """Anything that proposes k candidate assignments for a problem."""

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        ...


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

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        out: List[List[float]] = []
        for _ in range(k):
            trace = refine(
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
            out.append(trace.x)
        return out


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
            self.denoiser.load_state_dict(state)
        elif not checkpoint:
            warnings.warn(
                "LearnedSolver running with an UNTRAINED denoiser (no checkpoint). "
                "Results are not meaningful — set MARC_CKPT to Quang's checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.denoiser.eval()

    def sample(self, problem: Any, k: int) -> List[List[float]]:
        cas_engine = _cas_engine_for(problem.graph)
        out: List[List[float]] = []
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
                    continue
                cand = [float(v) for v in x.squeeze(-1).reshape(-1).tolist()]
                if self.polish:
                    cand = self._polish(problem.graph, cand)
                out.append(cand)
        return out

    def _polish(self, graph: Any, x0: List[float]) -> List[float]:
        """Diffusion proposes, energy-descent disposes: refine the (possibly coarse
        or diverged) diffusion candidate to strict tolerance. Reuses the same
        Langevin refinement as the baseline solver, seeded at the diffusion output."""
        from marc.refine.iterative import refine

        # guard against non-finite diffusion outputs before handing to the polisher
        if not all(abs(v) < 1e6 for v in x0):
            x0 = [0.0 for _ in x0]
        trace = refine(graph, x0, noise=True, seed=0)
        return trace.x


def load_solver(name: str | None = None, **kwargs: Any) -> Solver:
    """Return a solver by name, honouring env config with a graceful, loud fallback.

    Names: ``"refine"`` (default, real gradient solver), ``"dummy"`` (oracle-ish
    sanity solver), ``"learned"``/``"davin"`` (the diffusion ``solve()`` + a
    checkpoint). ``None`` reads ``MARC_SOLVER`` (default ``"refine"``). The learned
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

    raise ValueError(f"unknown solver '{name}' (expected refine|dummy|learned)")
