"""P3 — structure diffusion (Davin, milestone task, preliminary).

Value diffusion (P1) denoises node *values* on a fixed graph. Structure diffusion
denoises the graph's *topology*: whether each slot exists at all. We use the padded-
slot / D3PM formulation — a fixed pool of ``n_slots``, each carrying a categorical
type that can be ABSENT (the slot is not part of this problem) or an active type.

This package is the corruption/representation half of P3, plus the U5 trained
structure-invention policy:
    schema.py         — PaddedGraph + slot-type vocabulary (ABSENT at index 0)
    diffusion.py      — categorical forward corruption toward ABSENT (absorbing D3PM)
    invention_data.py — menu-based invention instances (fixed graph + K candidates)
    policy.py         — trained policy: encoder + StructureHead + D3PM reverse sampler

The reverse denoiser head lives in ``marc.model.structure_head`` (Quang). The pilot
that wires them together is ``scripts/train_structure_pilot.py``; the trained policy
harness is ``scripts/train_structure_policy.py`` / ``scripts/run_invention_eval.py``.
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
from .invention_data import (
    Candidate,
    InventionInstance,
    build_menu,
    make_dataset,
    to_padded,
)
from .policy import (
    M_MAX,
    SLOT_FEATURE_DIM,
    StructureEncoder,
    StructurePolicy,
    chosen_candidate,
    reverse_sample,
    slot_features,
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
    "Candidate",
    "InventionInstance",
    "build_menu",
    "make_dataset",
    "to_padded",
    "M_MAX",
    "SLOT_FEATURE_DIM",
    "StructureEncoder",
    "StructurePolicy",
    "chosen_candidate",
    "reverse_sample",
    "slot_features",
]
