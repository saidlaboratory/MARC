"""Categorical forward corruption for structure diffusion (P3 preliminary).

Implements the *forward* (noising) process over slot types — the discrete analogue of
adding Gaussian noise to values. We use an **absorbing D3PM** process: as ``t`` grows,
each slot type independently decays toward ``ABSENT``. Denoising (Quang's
``StructureHead``) learns the reverse — repopulating slots that should be active,
which is the "add an auxiliary node" behaviour P3 tests.

Marginal q(c_t | c_0) (absorbing):
    with probability   alpha_bar[t]   keep c_0
    with probability 1-alpha_bar[t]   set to ABSENT

``alpha_bar[t]`` is the cumulative keep-probability, 1.0 at t=0 (identity) decaying to
~0 at t=T-1 (everything absorbed). A ``"uniform"`` mode (flip to a random type) is also
provided — the spec allows "simple flip noise".
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph


def keep_schedule(T: int, kind: str = "cosine") -> torch.Tensor:
    """Cumulative keep-probability alpha_bar over t = 0..T-1 (1.0 -> ~0.0).

    Args:
        T:    number of diffusion steps.
        kind: "cosine" (default) or "linear".
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    t_norm = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)
    if kind == "cosine":
        return torch.cos(t_norm * (math.pi / 2.0)) ** 2
    if kind == "linear":
        return 1.0 - t_norm
    raise ValueError(f"unknown schedule kind: {kind!r}")


def corrupt_types(
    types: torch.Tensor,
    t: int,
    T: int,
    *,
    mode: str = "absorbing",
    schedule: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample noised slot types c_t ~ q(c_t | c_0=types) at step ``t``.

    Args:
        types:     LongTensor [n_slots] clean slot types (c_0).
        t:         diffusion step index in [0, T-1].
        T:         total steps.
        mode:      "absorbing" (decay toward ABSENT) or "uniform" (flip to random type).
        schedule:  optional precomputed ``keep_schedule(T)`` (avoids recompute in loops).
        generator: optional torch RNG for reproducibility.

    Returns:
        LongTensor [n_slots] noised slot types.
    """
    if not 0 <= t < T:
        raise ValueError(f"t={t} out of range [0, {T})")
    types = torch.as_tensor(types, dtype=torch.long).reshape(-1)
    if schedule is None:
        schedule = keep_schedule(T)
    keep_p = schedule[t]

    rand = torch.rand(types.shape, generator=generator)
    keep = rand < keep_p

    if mode == "absorbing":
        replacement = torch.full_like(types, ABSENT)
    elif mode == "uniform":
        replacement = torch.randint(
            0, NUM_SLOT_TYPES, types.shape, generator=generator, dtype=torch.long
        )
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return torch.where(keep, types, replacement)


def corrupt(
    graph: PaddedGraph,
    t: int,
    T: int,
    *,
    mode: str = "absorbing",
    schedule: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> PaddedGraph:
    """Corrupt a whole ``PaddedGraph``: noise the types and zero out newly-ABSENT slots.

    Values of slots that remain active are preserved; slots corrupted to ABSENT lose
    their value (an absent slot has no magnitude).
    """
    noised_types = corrupt_types(
        graph.slot_types, t, T, mode=mode, schedule=schedule, generator=generator
    )
    active = noised_types != ABSENT
    noised_values = torch.where(active, graph.values, torch.zeros_like(graph.values))
    return PaddedGraph(noised_types, noised_values)


def absent_fraction(types: torch.Tensor) -> float:
    """Fraction of slots that are ABSENT — the natural "how corrupted" scalar."""
    types = torch.as_tensor(types, dtype=torch.long).reshape(-1)
    if types.numel() == 0:
        return 0.0
    return float((types == ABSENT).float().mean().item())
