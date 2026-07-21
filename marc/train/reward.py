"""P2 — reward function for Stage-B GRPO (Davin, milestone task).

This module defines *how a denoising rollout is scored*. It is the reward half of
the P2 modeling/diffusion deliverable; the rollout half lives in
``marc.train.rollout``. Quang's GRPO trainer (``marc/train/stage_b.py``) imports
``compute_reward`` and calls it once per sampled trajectory.

Reward definition (TECHNICAL_GUIDE §8.2)
----------------------------------------
    R(o) = B * 1[checker accepts x_final]  +  shaping(o)

Terminal term
    The *conservative* two-stage ``Checker`` (numeric pre-filter + exact symbolic
    gate, ``marc.cas.checker``) is the authoritative acceptance test. This matters:
    a false-accept here poisons GRPO by handing full reward to a wrong solution
    ("a false-accepting checker is the most dangerous bug in RL"). The numeric-only
    ``CASEngine.accepts`` is *not* used as the gate — only, optionally, as guidance.

Shaping term — potential-based, optimum-preserving
    shaping(o) = sum_k ( E(x_{k-1}) - E(x_k) )  =  E(x_0) - E(x_final)

    With potential Phi(s) = -E(s) and gamma = 1, the per-step shaping
    F(s, s') = gamma*Phi(s') - Phi(s) = E(s) - E(s') telescopes to
    E(x_0) - E(x_final). This is Ng et al. (1999) potential-based shaping, which
    provably leaves the optimal policy unchanged. It is computed from the
    per-state energies recorded by ``run_rollout`` in ``trajectory["energy_trajectory"]``.

    NOTE: this is deliberately *not* ``-E(x_final)``. Because each GRPO rollout
    starts from an independent random x_0, the E(x_0) term is not constant across a
    group, so ``-E(x_final)`` and the potential-based form give different
    group-relative advantages. The potential-based form is the correct one.

Public API (stable — import without changes):
    compute_reward(trajectory, G, checker, cas, *, B=10.0, use_shaping=True,
                   shaping_clip=100.0) -> float
    terminal_reward(trajectory, G, checker, *, B=10.0) -> float
    shaping_reward(trajectory, *, cas=None) -> float

Trajectory contract (produced by ``marc.train.rollout.run_rollout``):
    x_final           : final variable assignment (tensor [n,1], array, or list)
    x_values          : optional list[float] convenience copy of x_final
    energy_trajectory : list[float] — E(x) at each visited state in temporal
                        (high-noise -> low-noise) order; len == steps + 1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Default terminal reward magnitude B (the checker-acceptance bonus).
TERMINAL_B: float = 10.0

# Symmetric bound on the shaping term inside compute_reward. Energy is a raw
# 0.5*sum(residual^2) on unnormalized values, so one diverging rollout (x clamped
# at +-1e4) can push the delta to O(1e16) and swamp the group-relative advantage.
SHAPING_CLIP: float = 10.0 * TERMINAL_B


def _x_values(trajectory: Dict[str, Any]) -> List[float]:
    """Extract the final assignment as a flat list[float], tolerating tensors/arrays."""
    vals: Any = trajectory.get("x_values")
    if vals is None:
        vals = trajectory["x_final"]
        squeeze = getattr(vals, "squeeze", None)
        if callable(squeeze):
            vals = squeeze()
        tolist = getattr(vals, "tolist", None)
        if callable(tolist):
            vals = tolist()
    if isinstance(vals, (float, int)):
        vals = [vals]
    return [float(v) for v in vals]


def shaping_reward(trajectory: Dict[str, Any], *, cas: Optional[Any] = None) -> float:
    """Potential-based shaping F = E(x_0) - E(x_final) from recorded per-state energies.

    Positive when energy decreased over the rollout (progress toward the constraint
    surface). Requires ``trajectory["energy_trajectory"]`` with >= 2 entries. If those
    were not recorded, degrades to the single-state potential ``-E(x_final)`` when a
    ``cas`` engine is supplied, else 0.0.
    """
    energies = trajectory.get("energy_trajectory") or []
    if len(energies) >= 2:
        return float(energies[0] - energies[-1])
    if cas is not None:
        return float(-cas.energy(_x_values(trajectory)))
    return 0.0


def terminal_reward(
    trajectory: Dict[str, Any],
    G: Any,
    checker: Any,
    *,
    B: float = TERMINAL_B,
) -> float:
    """B if the conservative checker accepts x_final, else 0.0.

    ``checker`` is expected to be a ``marc.cas.checker.Checker`` (exact/symbolic).
    If ``checker`` is None, there is no authoritative gate and the terminal reward is
    0.0 — callers must pass a real checker to earn terminal reward.
    """
    if checker is None:
        return 0.0
    x = _x_values(trajectory)
    return float(B) if checker.accepts(G, x) else 0.0


def compute_reward(
    trajectory: Dict[str, Any],
    G: Any,
    checker: Any,
    cas: Any,
    *,
    B: float = TERMINAL_B,
    use_shaping: bool = True,
    shaping_clip: float = SHAPING_CLIP,
) -> float:
    """Total scalar reward for one denoising rollout.

    Args:
        trajectory: rollout dict from ``run_rollout`` (see module contract).
        G:          the ``FactorGraph`` the rollout solved.
        checker:    conservative ``Checker`` — authoritative acceptance gate (terminal).
        cas:        ``CASEngine`` — exact energy, used for shaping fallback only.
        B:          terminal reward magnitude for a checker-accepted solution.
        use_shaping: include potential-based energy shaping (set False for the
                     "purist" checker-only ablation).
        shaping_clip: symmetric bound on the shaping term; keeps a single
                     diverged rollout from dominating the GRPO group std.

    Returns:
        float reward R = terminal + (clipped shaping if use_shaping).
    """
    reward = terminal_reward(trajectory, G, checker, B=B)
    if use_shaping:
        shaping = shaping_reward(trajectory, cas=cas)
        reward += max(-shaping_clip, min(shaping_clip, shaping))
    return float(reward)
