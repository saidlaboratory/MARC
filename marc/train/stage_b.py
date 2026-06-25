"""Stage B: GRPO reinforcement-learning fine-tuning of the denoising policy.

The diffusion reverse process is treated as a finite-horizon MDP:
  State  : (x_k, G) — current variable values + factor graph
  Action : one denoising step sampled from N(eps_hat, I)
  Reward : terminal checker reward B + energy-shaping per step

GRPO optimizer: sample N rollouts per problem, compute group-relative
advantage, optimise clipped surrogate + KL leash to reference policy.

    R = B * I[checker accepts x_K] + sum_k (E(x_{k-1}) - E(x_k))

    J = E[min(rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i) - beta KL(pi||pi_ref)]
    where
        rho_i = pi_theta(o_i|q) / pi_theta_old(o_i|q)
        A_i   = (R_i - mean(R)) / (std(R) + 1e-8)

Gradient flow note:
The policy is modelled as N(eps_hat(x, t; theta), I) over the noise
residual.  The sampled action at each step is:
    epsilon_sample = eps_hat + z,   z ~ N(0, I)
so
    log p_theta(epsilon_sample | x, t) = -0.5 ||z||^2  +  const
and the gradient w.r.t. theta flows through eps_hat via the squared
residual  ||epsilon_sample - eps_hat||^2 / 2.
"""

import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Precomputed constant used in every Gaussian log-prob evaluation.
_LOG_2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def sample_trajectory(
    policy: nn.Module,
    data,
    n_vars: int,
    alpha_bar: torch.Tensor,
    steps: int = 40,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Sample one stochastic denoising trajectory without gradients.

    The policy is treated as predicting the *mean* eps_hat of a Gaussian
    action distribution N(eps_hat, I).  At each step we draw:

        z ~ N(0, I)
        epsilon_sample = eps_hat + z       <- actual noise used in DDIM step
        log p(epsilon_sample | x, t; theta) = -0.5 ||z||^2  + const(n_vars)

    The stored ``eps_hats`` and ``raw_noises`` (z values) allow
    ``_compute_log_prob_with_grad`` to recompute differentiable log-probs
    for the policy-gradient update.

    Returns:
        "x_final"         : [n_vars, 1] tensor — final variable assignment
        "log_prob"        : scalar tensor — sum of log-probs (no grad)
        "energy_trajectory": list[float]
        "eps_hats"        : list of detached [n_vars, 1] tensors per step
        "raw_noises"      : list of [n_vars, 1] tensors (z) per step
    """
    policy.eval()
    data = data.to(device)
    x = torch.randn(n_vars, 1, device=device)

    total_log_prob = torch.zeros((), device=device)
    energy_traj: List[float] = []
    eps_hats: List[torch.Tensor] = []
    raw_noises: List[torch.Tensor] = []

    with torch.no_grad():
        for step in reversed(range(steps)):
            t = torch.tensor([step + 1], device=device)
            eps_hat = policy(data, t).detach()  # [n_vars, 1]
            eps_hats.append(eps_hat)

            abar_t = alpha_bar[step]
            abar_prev = alpha_bar[step - 1] if step > 0 else torch.ones(1, device=device)

            # Predicted x0 from current x and eps_hat
            x0_pred = (x - (1 - abar_t).sqrt() * eps_hat) / abar_t.sqrt()
            x0_pred = x0_pred.clamp(-10, 10)

            sigma_t = ((1 - abar_prev) / (1 - abar_t)).sqrt() * (1 - abar_t / abar_prev).sqrt()

            # Action: sample z ~ N(0, I) and add to eps_hat
            z = torch.randn_like(x)
            raw_noises.append(z)
            epsilon_sample = eps_hat + z

            # Log-prob under N(eps_hat, I):  -0.5 ||z||^2 + const
            log_p = -0.5 * (z ** 2).sum() - 0.5 * n_vars * _LOG_2PI
            total_log_prob = total_log_prob + log_p

            # DDIM step using epsilon_sample
            direction = (1 - abar_prev - sigma_t ** 2).clamp(min=0).sqrt() * epsilon_sample
            x = abar_prev.sqrt() * x0_pred + direction + sigma_t * z

    return {
        "x_final": x.detach(),
        "log_prob": total_log_prob.detach(),
        "energy_trajectory": energy_traj,
        "eps_hats": eps_hats,
        "raw_noises": raw_noises,
    }


def _compute_log_prob_with_grad(
    policy: nn.Module,
    data,
    n_vars: int,
    epsilon_samples: List[torch.Tensor],
    steps: int,
    device: str,
) -> torch.Tensor:
    """Re-compute trajectory log-prob under current policy WITH gradients.

    The policy outputs eps_hat, the *mean* of a Gaussian action distribution
    N(eps_hat, I).  The sampled actions (epsilon_samples = eps_hat_old + z)
    are fixed from the rollout; gradient flows through the new eps_hat via:

        log p_theta(epsilon_sample) = -0.5 ||epsilon_sample - eps_hat_new||^2 + const
    """
    policy.train()
    data = data.to(device)

    total_log_prob = torch.zeros((), device=device)
    eps_iter = iter(epsilon_samples)

    for step in reversed(range(steps)):
        t = torch.tensor([step + 1], device=device)
        eps_hat_new = policy(data, t)  # grad enabled

        residual = next(eps_iter) - eps_hat_new
        log_p = -0.5 * (residual ** 2).sum() - 0.5 * n_vars * _LOG_2PI
        total_log_prob = total_log_prob + log_p

    return total_log_prob


def _epsilon_samples(traj: Dict[str, Any]) -> List[torch.Tensor]:
    """Return the list of sampled actions: epsilon = eps_hat_old + z."""
    return [eh + z for eh, z in zip(traj["eps_hats"], traj["raw_noises"])]


def compute_reward(
    trajectory: Dict[str, Any],
    cas_engine,
    variable_ids: List[str],
    B: float = 10.0,
    use_energy_shaping: bool = True,
) -> float:
    """Compute reward for a trajectory.

    R = B * I[checker accepts x_final] + shaping
    shaping = -E(x_final)  (reward low-energy solutions)
    """
    x_final = trajectory["x_final"]
    x_vals = x_final.squeeze().tolist()
    if isinstance(x_vals, float):
        x_vals = [x_vals]

    terminal_reward = B if cas_engine.accepts(x_vals) else 0.0

    if use_energy_shaping:
        final_energy = cas_engine.energy(x_vals)
        return terminal_reward + (-final_energy)

    return terminal_reward


def grpo_step(
    policy: nn.Module,
    ref_policy: Optional[nn.Module],
    data,
    n_vars: int,
    cas_engine,
    variable_ids: List[str],
    alpha_bar: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    N: int = 8,
    B: float = 10.0,
    beta: float = 0.01,
    eps_clip: float = 0.2,
    steps: int = 40,
    device: str = "cpu",
    purist: bool = False,
) -> Dict[str, float]:
    """One GRPO update step for a single problem.

    1. Sample N rollouts (no grad) — records epsilon_samples per step
    2. Compute rewards R_1,...,R_N
    3. Compute group-relative advantage A_i = (R_i - mean(R)) / (std(R) + 1e-8)
    4. Re-compute log p_theta(epsilon_samples) WITH grad via Gaussian likelihood
    5. Clipped PPO surrogate + KL penalty
    6. Backprop and step
    """
    policy.train()
    data = data.to(device)

    # --- Phase 1: rollouts (no grad) ---
    trajectories = [
        sample_trajectory(policy, data, n_vars, alpha_bar, steps, device)
        for _ in range(N)
    ]

    rewards = torch.tensor(
        [
            compute_reward(traj, cas_engine, variable_ids, B=B, use_energy_shaping=not purist)
            for traj in trajectories
        ],
        dtype=torch.float,
        device=device,
    )

    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r

    # --- Phase 2: differentiable log-probs under current policy ---
    new_log_probs = torch.stack([
        _compute_log_prob_with_grad(
            policy, data, n_vars, _epsilon_samples(traj), steps, device,
        )
        for traj in trajectories
    ])

    old_log_probs = torch.stack([traj["log_prob"] for traj in trajectories])

    # Clipped surrogate (PPO/GRPO style)
    rhos = torch.exp(new_log_probs - old_log_probs.detach())
    surr1 = rhos * advantages
    surr2 = rhos.clamp(1 - eps_clip, 1 + eps_clip) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()

    # KL penalty against frozen reference policy
    if ref_policy is not None:
        with torch.no_grad():
            ref_trajs = [
                sample_trajectory(ref_policy, data, n_vars, alpha_bar, steps, device)
                for _ in range(N)
            ]
        # log-probs under ref policy, detached (no grad through ref)
        ref_log_probs = torch.stack([
            _compute_log_prob_with_grad(
                ref_policy, data, n_vars, _epsilon_samples(rt), steps, device,
            ).detach()
            for rt in ref_trajs
        ])
        kl = (new_log_probs - ref_log_probs).mean()
    else:
        kl = torch.tensor(0.0, device=device)

    loss = pg_loss + beta * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accept_count = sum(1 for r in rewards if r >= B * 0.9)
    return {
        "loss": loss.item(),
        "mean_reward": mean_r.item(),
        "accept_rate": accept_count / N,
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
) -> nn.Module:
    """Full Stage B GRPO training loop.

    Iterates over problems, applying grpo_step for each.
    Saves epoch checkpoints to checkpoint_dir.

    Args:
        policy: the model to fine-tune
        ref_policy: frozen Stage-A reference model (None to skip KL leash)
        problems: list of (FactorGraph, solution_dict, CASEngine) tuples
        alpha_bar: cumulative alpha schedule [T]
        epochs: number of training epochs
        N: rollouts per problem
        B: terminal reward magnitude
        beta: KL penalty coefficient
        lr: Adam learning rate
        checkpoint_dir: directory to save epoch checkpoints
        device: torch device string
        purist: if True, use terminal reward only (no energy shaping)
    """
    from marc.graph.pyg import build_heterodata

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy = policy.to(device)

    if ref_policy is not None:
        ref_policy = ref_policy.to(device)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    for epoch in range(epochs):
        epoch_stats = []
        for graph, _solution, cas_engine in problems:
            data = build_heterodata(graph)
            variable_ids = [v.id for v in graph.variables]
            n_vars = len(variable_ids)

            stats = grpo_step(
                policy, ref_policy, data, n_vars, cas_engine, variable_ids,
                alpha_bar, optimizer, N=N, B=B, beta=beta,
                steps=20, device=device, purist=purist,
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
