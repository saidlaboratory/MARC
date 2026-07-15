"""P3 — structure diffusion (Davin, milestone task, preliminary).

Value diffusion (P1) denoises node *values* on a fixed graph. Structure diffusion
denoises the graph's *topology*: whether each slot exists at all. We use the padded-
slot / D3PM formulation — a fixed pool of ``n_slots``, each carrying a categorical
type that can be ABSENT (the slot is not part of this problem) or an active type.

This package is the corruption/representation half of P3:
    schema.py    — PaddedGraph + slot-type vocabulary (ABSENT at index 0)
    diffusion.py — categorical forward corruption toward ABSENT (absorbing D3PM)

The reverse denoiser head lives in ``marc.model.structure_head`` (Quang). The pilot
that wires them together is ``scripts/train_structure_pilot.py``.
"""

from .schema import (
    ABSENT,
    NUM_SLOT_TYPES,
    PaddedGraph,
    SlotType,
)
from .diffusion import (
    absent_fraction,
    corrupt,
    corrupt_types,
    keep_schedule,
)

__all__ = [
    "ABSENT",
    "NUM_SLOT_TYPES",
    "PaddedGraph",
    "SlotType",
    "absent_fraction",
    "corrupt",
    "corrupt_types",
    "keep_schedule",
]
