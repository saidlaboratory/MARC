"""Padded-slot schema for structure diffusion (P3 preliminary).

A ``PaddedGraph`` is a fixed pool of ``n_slots`` variable slots. Each slot has:
  - a categorical **type** in ``SlotType`` — ``ABSENT`` (index 0) means the slot is
    not part of the problem; any other type means the slot is active/present;
  - a scalar **value** (meaningful only for active slots; ABSENT slots carry 0).

Fixing the slot count sidesteps the hard "grow the graph" problem: adding an
auxiliary node = flipping a slot from ABSENT to active. ``ABSENT`` is index 0 to match
``marc.model.structure_head.StructureHead.ABSENT_TYPE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch


class SlotType(IntEnum):
    """Slot-type vocabulary. Index 0 MUST be ABSENT (matches StructureHead)."""

    ABSENT = 0
    VARIABLE = 1


#: Integer type index for an empty slot (the D3PM absorbing state).
ABSENT: int = int(SlotType.ABSENT)

#: Size of the categorical slot-type vocabulary.
NUM_SLOT_TYPES: int = len(SlotType)


@dataclass
class PaddedGraph:
    """A fixed-size pool of variable slots for structure diffusion.

    Attributes:
        slot_types: LongTensor [n_slots] — per-slot type (0 == ABSENT).
        values:     FloatTensor [n_slots] — per-slot value (0 for ABSENT slots).
    """

    slot_types: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        self.slot_types = torch.as_tensor(self.slot_types, dtype=torch.long).reshape(-1)
        self.values = torch.as_tensor(self.values, dtype=torch.float32).reshape(-1)
        if self.slot_types.shape != self.values.shape:
            raise ValueError(
                f"slot_types {tuple(self.slot_types.shape)} and values "
                f"{tuple(self.values.shape)} must have the same length"
            )

    # --- construction ------------------------------------------------------

    @classmethod
    def from_active(
        cls,
        active_values: Sequence[float],
        n_slots: int,
        slot_type: int = int(SlotType.VARIABLE),
    ) -> "PaddedGraph":
        """Pad a list of active variable values into an ``n_slots`` pool.

        The first ``len(active_values)`` slots become active (``slot_type``) with the
        given values; the remaining slots are ABSENT with value 0.
        """
        k = len(active_values)
        if k > n_slots:
            raise ValueError(f"{k} active values do not fit in {n_slots} slots")
        types = torch.full((n_slots,), ABSENT, dtype=torch.long)
        vals = torch.zeros(n_slots, dtype=torch.float32)
        if k:
            types[:k] = slot_type
            vals[:k] = torch.as_tensor(active_values, dtype=torch.float32)
        return cls(types, vals)

    # --- queries -----------------------------------------------------------

    @property
    def n_slots(self) -> int:
        return int(self.slot_types.shape[0])

    def active_mask(self) -> torch.Tensor:
        """Bool [n_slots] — True where the slot is present (non-ABSENT)."""
        return self.slot_types != ABSENT

    def num_active(self) -> int:
        return int(self.active_mask().sum().item())

    def clone(self) -> "PaddedGraph":
        return PaddedGraph(self.slot_types.clone(), self.values.clone())

    # --- features ----------------------------------------------------------

    def to_features(self) -> torch.Tensor:
        """Per-slot input features for a denoiser encoder.

        Returns FloatTensor [n_slots, NUM_SLOT_TYPES + 1]:
        one-hot(slot_type) concatenated with the (present-only) value. ABSENT slots
        contribute a zero value so the encoder cannot read a "ghost" magnitude.
        """
        onehot = torch.nn.functional.one_hot(
            self.slot_types, num_classes=NUM_SLOT_TYPES
        ).float()
        present_values = torch.where(
            self.active_mask(), self.values, torch.zeros_like(self.values)
        )
        return torch.cat([onehot, present_values.unsqueeze(-1)], dim=-1)
