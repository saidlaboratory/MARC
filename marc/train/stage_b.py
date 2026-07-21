"""Stage B: GRPO reinforcement-learning fine-tuning of the denoising policy.

The diffusion reverse process is treated as a finite-horizon MDP:
  State  : (x_k, G) — current variable values + factor graph
  Action : one denoising step sampled from N(eps_hat, I)
  Reward : terminal checker reward B + potential-based energy shaping

GRPO update, per problem (TECHNICAL_GUIDE §8.2):
  1. Sample N trajectories ONCE via ``marc.train.rollout.run_rollout`` — it
     records per-step states, actions, the behavioral log-prob, per-state
     energies, and the checker verdict.
  2. Score each with ``marc.train.reward.compute_reward`` (the conservative
     ``Checker`` is the authoritative terminal gate); group-normalize:
         A_i = (R_i - mean(R)) / (std(R) + 1e-8)
  3. new_lp_i = ``recompute_log_prob`` under the current policy — replays the
     RECORDED per-step states, so the policy is conditioned on x_k at every
     step (the earlier inline version fed a static input to all steps, which
     made the gradient meaningless).
  4. old_lp_i = the log-prob recorded at sampling time (frozen behavioral term).
  5. KL leash: ref-policy log-prob of the SAME trajectories, cached under
     no_grad — no separate ref rollouts (the earlier version re-sampled ref
     trajectories and compared log-probs of different trajectories: a
     meaningless KL at 2x the compute).
  6. Clipped surrogate + beta * KL, gradient clipping, Adam step:
         rho_i = exp(clamp(new_lp_i - old_lp_i, -20, 20))
         J = E[min(rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i)] - beta KL

Gradient flow note:
The policy is N(eps_hat(x_k, t; theta), I) over the noise residual, so
    log p_theta(epsilon | x_k, t) = -0.5 ||epsilon - eps_hat||^2 + const
and the gradient w.r.t. theta flows through eps_hat.
"""

import os
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from marc.cas.checker import Checker
from marc.train.reward import SHAPING_CLIP, compute_reward
from marc.train.rollout import recompute_log_prob, run_rollout


def grpo_step(
    policy: nn.Module,
    ref_policy: Optional[nn.Module],
    G,
    cas_engine,
    alpha_bar: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    checker=None,
    N: int = 8,
    B: float = 10.0,
    beta: float = 0.01,
    eps_clip: float = 0.2,
    steps: int = 40,
    device: str = "cpu",
    purist: bool = False,
    grad_clip: float = 1.0,
    entropy_coef: float = 0.0,
    shaping_clip: float = SHAPING_CLIP,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """One GRPO update for a single ``FactorGraph`` problem.

    Args:
        policy:      the model being fine-tuned.
        ref_policy:  frozen reference model for the KL leash (None to skip).
        G:           the ``FactorGraph`` to solve (rollouts build their own data).
        cas_engine:  ``CASEngine`` — per-state energy for shaping (skipped when
                     ``purist``, where shaping is unused).
        alpha_bar:   cumulative alpha schedule [T]; its length sets schedule_T.
        optimizer:   optimizer over ``policy.parameters()``.
        checker:     conservative ``Checker`` — authoritative terminal gate and
                     source of the ``accept_rate`` stat (None → no terminal reward).
        N:           rollouts per update (the GRPO group).
        B:           terminal reward magnitude.
        beta:        KL penalty coefficient.
        eps_clip:    PPO clip range for the importance ratio.
        steps:       denoising steps per rollout (episode horizon K).
        purist:      terminal reward only — no energy shaping, no energy evals.
        grad_clip:   max gradient norm.
        entropy_coef: accepted for API completeness; adds NO term (see below).
        shaping_clip: symmetric bound on the shaping reward (see compute_reward).
        generator:   optional torch.Generator for reproducible rollouts.

    Returns:
        {"loss", "pg_loss", "kl", "mean_reward", "accept_rate", "grad_norm"}
    """
    if entropy_coef < 0:
        raise ValueError(f"entropy_coef must be >= 0, got {entropy_coef}")
    if entropy_coef:
        # ponytail: entropy const under fixed variance; becomes real only if sigma becomes learnable
        warnings.warn(
            "entropy_coef has no effect: the policy is N(eps_hat, I) with fixed "
            "variance, so its entropy does not depend on theta.",
            stacklevel=2,
        )

    # --- 1. Sampling pass, ONCE (no grad; behavioral policy) ---
    trajectories = [
        run_rollout(
            policy,
            G,
            K=steps,
            cas=None if purist else cas_engine,
            checker=checker,
            schedule_T=alpha_bar.numel(),
            device=device,
            generator=generator,
        )
        for _ in range(N)
    ]

    # --- 2. Rewards and group-relative advantages ---
    rewards = torch.tensor(
        [
            compute_reward(traj, G, checker, cas_engine, B=B,
                           use_shaping=not purist, shaping_clip=shaping_clip)
            for traj in trajectories
        ],
        dtype=torch.float,
        device=device,
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # --- 3/4. Log-probs: recorded behavioral vs. current policy on replayed states ---
    old_lp = torch.stack([traj["log_prob"] for traj in trajectories]).detach()
    new_lp = torch.stack(
        [recompute_log_prob(policy, G, traj, device=device) for traj in trajectories]
    )

    # --- 5. KL leash on the SAME trajectories (cached; no ref rollouts) ---
    if ref_policy is not None:
        with torch.no_grad():
            ref_lp = torch.stack(
                [recompute_log_prob(ref_policy, G, traj, device=device) for traj in trajectories]
            )
        kl = (new_lp - ref_lp).mean()
    else:
        kl = torch.zeros((), device=device)

    # --- 6. Clipped surrogate. log_ratio sums steps*n Gaussian terms — clamp
    # before exp so a large policy move can't overflow rho. ---
    log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
    rho = log_ratio.exp()
    pg_loss = -torch.min(
        rho * advantages,
        rho.clamp(1 - eps_clip, 1 + eps_clip) * advantages,
    ).mean()
    loss = pg_loss + beta * kl

    # --- 7. Update ---
    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    accept_rate = sum(1 for traj in trajectories if traj["accepted"]) / N
    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl": kl.item(),
        "mean_reward": rewards.mean().item(),
        "accept_rate": accept_rate,
        "grad_norm": float(grad_norm),
    }


def train_stage_b(
    policy: nn.Module,
    ref_policy: Optional[nn.Module],
    problems,
    alpha_bar: torch.Tensor,
    epochs: int = 5,
    N: int = 8,
    B: float = 10.0,
    beta: float = 0.01,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints/stage_b",
    device: str = "cpu",
    purist: bool = False,
    *,
    steps: int = 20,
    grad_clip: float = 1.0,
    seed: Optional[int] = None,
    entropy_coef: float = 0.0,
    eps_clip: float = 0.2,
    shaping_clip: float = SHAPING_CLIP,
    checker=None,
) -> nn.Module:
    """Full Stage B GRPO training loop.

    Iterates over problems, applying ``grpo_step`` for each.
    Saves epoch checkpoints to checkpoint_dir.

    Args:
        policy: the model to fine-tune
        ref_policy: frozen Stage-A reference model (None to skip KL leash)
        problems: list of (FactorGraph, solution_or_None, CASEngine) tuples
        alpha_bar: cumulative alpha schedule [T]
        epochs: number of training epochs
        N: rollouts per problem
        B: terminal reward magnitude
        beta: KL penalty coefficient
        lr: Adam learning rate
        checkpoint_dir: directory to save epoch checkpoints
        device: torch device string
        purist: if True, use terminal reward only (no energy shaping)
        steps: denoising steps per rollout (20 preserves the historical default)
        grad_clip: max gradient norm per update
        seed: if set, seeds torch globally and each rollout group deterministically
        entropy_coef: forwarded to grpo_step (no-op; see grpo_step)
        eps_clip: PPO clip range
        shaping_clip: symmetric bound on the shaping reward (see compute_reward)
        checker: authoritative terminal gate; defaults to ``Checker()``
    """
    checker = checker or Checker()
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy = policy.to(device)

    if ref_policy is not None:
        ref_policy = ref_policy.to(device)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    if seed is not None:
        torch.manual_seed(seed)

    for epoch in range(epochs):
        epoch_stats = []
        for problem_idx, (graph, _solution, cas_engine) in enumerate(problems):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed + epoch * 1_000_003 + problem_idx * 97)

            stats = grpo_step(
                policy, ref_policy, graph, cas_engine, alpha_bar, optimizer,
                checker=checker, N=N, B=B, beta=beta, eps_clip=eps_clip,
                steps=steps, device=device, purist=purist, grad_clip=grad_clip,
                entropy_coef=entropy_coef, shaping_clip=shaping_clip,
                generator=generator,
            )
            epoch_stats.append(stats)

        n = len(epoch_stats)
        avg_loss = sum(s["loss"] for s in epoch_stats) / n if n else 0.0
        avg_reward = sum(s["mean_reward"] for s in epoch_stats) / n if n else 0.0
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}, mean_reward: {avg_reward:.4f}")

        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": avg_reward,
            },
            ckpt_path,
        )

    return policy
