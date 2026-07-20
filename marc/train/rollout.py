"""P2 — rollout interface for Stage-B GRPO (Davin, milestone task).

Wraps the reverse-diffusion *solve* steps into a trajectory (an MDP episode) that
Quang's GRPO trainer (``marc/train/stage_b.py``) can score with
``marc.train.reward.compute_reward`` and differentiate for the policy-gradient update.

Diffusion-as-policy (TECHNICAL_GUIDE §8.2)
------------------------------------------
At denoising step k the policy (the ``GraphDenoiser``) outputs the mean ``eps_hat``
of a Gaussian action distribution ``N(eps_hat, I)``. We sample ``z ~ N(0, I)``; the
action is ``epsilon = eps_hat + z`` and

    log p(epsilon | x_k, t; theta) = -0.5 * ||z||^2  -  (n/2) * log(2*pi)

The DDIM update is applied *deterministically* given the sampled ``epsilon`` (eta=0),
so the single, tractable source of stochasticity is the action noise ``z`` — exactly
what the log-prob accounts for. This keeps the policy-gradient estimator unbiased.

Two correctness properties this module guarantees (and the earlier inline sampler did not):
  1. ``energy_trajectory`` is actually populated — one ``E(x)`` per visited state —
     so ``marc.train.reward.shaping_reward`` (potential-based) is computable.
  2. The policy is conditioned on the *current* noisy values each step
     (``data["variable"].x`` is refreshed to x_k), so ``eps_hat`` actually depends on
     the state — which is required for the gradient in ``recompute_log_prob`` to be
     meaningful.

Public API (stable — import without changes):
    run_rollout(policy, G, K=40, *, cas=None, checker=None, ...) -> trajectory dict
    epsilon_samples(trajectory) -> list[Tensor]          # actions = eps_hat + z
    recompute_log_prob(policy, G, trajectory, ...) -> Tensor   # differentiable log-prob

Trajectory dict:
    x_final           : Tensor[n,1]                 final assignment
    x_values          : list[float]                 flat copy of x_final
    log_prob          : scalar Tensor (detached)    sum of per-step Gaussian log-probs
    energy_trajectory : list[float]                 E(x) per state, temporal order, len K+1
    eps_hats          : list[Tensor[n,1]]           detached policy means per step
    raw_noises        : list[Tensor[n,1]]           sampled z per step
    states            : list[Tensor[n,1]]           input x_k per step (for grad replay)
    step_indices      : list[int]                   schedule index t used at each step
    accepted          : bool | None                 checker verdict at x_final (if given)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from marc.diffusion.sample import ddim_step
from marc.diffusion.schedule import cosine_beta_schedule
from marc.graph.pyg import build_heterodata

_LOG_2PI = math.log(2.0 * math.pi)

# GRPO calls run_rollout N times per update; regenerating the schedule each call
# is pure waste. Read-only after creation, so sharing the tensor is safe.
_SCHEDULE_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _cached_alpha_bar(T: int, device: str = "cpu") -> torch.Tensor:
    """alpha_bar of the cosine schedule of length T, cached per (T, device)."""
    key = (T, str(device))
    if key not in _SCHEDULE_CACHE:
        _, alpha_bar = cosine_beta_schedule(T)
        _SCHEDULE_CACHE[key] = alpha_bar.to(device)
    return _SCHEDULE_CACHE[key]


def _energy(cas: Optional[Any], x: torch.Tensor) -> Optional[float]:
    if cas is None:
        return None
    return float(cas.energy(x.detach().reshape(-1).tolist()))


def _schedule_indices(K: int, T: int) -> List[int]:
    """Map K rollout steps onto schedule indices, high-noise (T-1) -> low-noise (0)."""
    idx = torch.linspace(T - 1, 0, K).round().long().tolist()
    return [int(i) for i in idx]


def run_rollout(
    policy: torch.nn.Module,
    G: Any,
    K: int = 40,
    *,
    cas: Optional[Any] = None,
    checker: Optional[Any] = None,
    use_guidance: bool = False,
    schedule_T: int = 1000,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Sample one stochastic denoising trajectory (no gradients).

    Args:
        policy:  the ``GraphDenoiser`` — ``policy(data, t[, cas])`` -> eps_hat [n,1].
        G:       the ``FactorGraph`` to solve.
        K:       number of denoising steps (episode horizon).
        cas:     ``CASEngine`` used to (a) record per-state energy and (b) feed
                 residual guidance to the policy when ``use_guidance``.
        checker: conservative ``Checker``; if given, records ``accepted`` at x_final.
        use_guidance: pass ``cas`` into the policy for CAS-guided denoising.
        schedule_T: length of the cosine noise schedule.
        device / generator: standard torch placement / reproducibility controls.

    Returns:
        trajectory dict (see module docstring).
    """
    n = len(G.variables)
    data = build_heterodata(G).to(device)
    alpha_bar = _cached_alpha_bar(schedule_T, device)
    step_indices = _schedule_indices(K, schedule_T)

    x = torch.randn(n, 1, device=device, generator=generator)

    eps_hats: List[torch.Tensor] = []
    raw_noises: List[torch.Tensor] = []
    states: List[torch.Tensor] = []
    energy_traj: List[float] = []

    total_log_prob = torch.zeros((), device=device)

    policy.eval()
    with torch.no_grad():
        e0 = _energy(cas, x)
        if e0 is not None:
            energy_traj.append(e0)

        for t_idx in step_indices:
            # Condition the policy on the CURRENT noisy values.
            data["variable"].x = x
            states.append(x.detach().clone())

            t = torch.tensor([t_idx], device=device)
            eps_hat = policy(data, t, cas) if use_guidance else policy(data, t)
            eps_hat = eps_hat.detach()
            eps_hats.append(eps_hat)

            # Action: epsilon = eps_hat + z, z ~ N(0, I).
            z = torch.randn(n, 1, device=device, generator=generator)
            raw_noises.append(z)
            epsilon = eps_hat + z

            log_p = -0.5 * (z ** 2).sum() - 0.5 * n * _LOG_2PI
            total_log_prob = total_log_prob + log_p

            # Deterministic DDIM update given the sampled epsilon.
            x = ddim_step(x, epsilon, t_idx, alpha_bar, eta=0.0)
            x = x.clamp(-1e4, 1e4)

            e = _energy(cas, x)
            if e is not None:
                energy_traj.append(e)

    x_values = x.detach().reshape(-1).tolist()
    accepted = checker.accepts(G, x_values) if checker is not None else None

    return {
        "x_final": x.detach(),
        "x_values": x_values,
        "log_prob": total_log_prob.detach(),
        "energy_trajectory": energy_traj,
        "eps_hats": eps_hats,
        "raw_noises": raw_noises,
        "states": states,
        "step_indices": step_indices,
        "accepted": accepted,
    }


def epsilon_samples(trajectory: Dict[str, Any]) -> List[torch.Tensor]:
    """Reconstruct the sampled actions epsilon = eps_hat + z from a trajectory."""
    return [eh + z for eh, z in zip(trajectory["eps_hats"], trajectory["raw_noises"])]


def recompute_log_prob(
    policy: torch.nn.Module,
    G: Any,
    trajectory: Dict[str, Any],
    *,
    cas: Optional[Any] = None,
    use_guidance: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Differentiable log-prob of a stored trajectory under the *current* policy.

    Replays the recorded states/timesteps, so gradient flows through the fresh
    ``eps_hat`` via ``log p = -0.5 * ||epsilon_sample - eps_hat_new||^2 + const``.
    This is what GRPO differentiates (the importance ratio rho_i uses it).
    """
    policy.train()
    data = build_heterodata(G).to(device)
    n = len(G.variables)

    actions = epsilon_samples(trajectory)
    total_log_prob = torch.zeros((), device=device)

    for x_k, t_idx, epsilon in zip(
        trajectory["states"], trajectory["step_indices"], actions
    ):
        data["variable"].x = x_k.to(device)
        t = torch.tensor([t_idx], device=device)
        eps_hat_new = policy(data, t, cas) if use_guidance else policy(data, t)
        residual = epsilon.to(device) - eps_hat_new
        total_log_prob = total_log_prob + (
            -0.5 * (residual ** 2).sum() - 0.5 * n * _LOG_2PI
        )

    return total_log_prob
