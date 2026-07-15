"""P3 pilot: structure-diffusion prototype on a toy (Davin, milestone).

Wires the categorical forward corruption (``marc.structure.diffusion``) to the reverse
denoiser head (``marc.model.structure_head.StructureHead``, Quang) and trains a tiny
encoder+head to recover clean slot types from ABSENT-corrupted ones. Then it runs the
reverse process from an all-ABSENT prior and checks the P3 done-condition:

    "One toy run shows a slot going ABSENT -> active."

Preliminary by design: 1 toy, <=1000 steps, CPU-only, seconds to run. Writes
``results/p3_structure/pilot_report.md``.

Usage:
    python scripts/train_structure_pilot.py [--steps 600] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Allow `python scripts/train_structure_pilot.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marc.model.structure_head import StructureHead
from marc.structure.diffusion import absent_fraction, corrupt, keep_schedule
from marc.structure.schema import ABSENT, NUM_SLOT_TYPES, PaddedGraph, SlotType

REPORT_PATH = os.path.join("results", "p3_structure", "pilot_report.md")


class SlotEncoder(nn.Module):
    """Tiny per-slot MLP encoder: features -> embeddings h_v for the StructureHead."""

    def __init__(self, in_dim: int, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


#: feature width = one-hot(type) ++ value ++ timestep ++ slot-position
FEATURE_DIM = NUM_SLOT_TYPES + 3


def _features(graph: PaddedGraph, t: int, T: int) -> torch.Tensor:
    """[n_slots, NUM_SLOT_TYPES + 3] = one-hot(type) ++ value ++ timestep ++ position.

    The slot-position column gives each slot a stable identity, breaking the
    permutation symmetry that would otherwise make an all-ABSENT prior unrecoverable
    (every slot would look identical and get the same prediction).
    """
    n = graph.n_slots
    base = graph.to_features()  # [n_slots, NUM_SLOT_TYPES + 1]
    t_col = torch.full((n, 1), t / max(T - 1, 1), dtype=torch.float32)
    pos_col = (torch.arange(n, dtype=torch.float32) / max(n - 1, 1)).unsqueeze(-1)
    return torch.cat([base, t_col, pos_col], dim=-1)


def make_toy() -> PaddedGraph:
    """A toy padded graph: 3 active variable slots in a pool of 8 (rest ABSENT)."""
    return PaddedGraph.from_active([2.0, 1.0, 3.0], n_slots=8, slot_type=int(SlotType.VARIABLE))


def train(
    steps: int = 600,
    T: int = 50,
    D: int = 32,
    lr: float = 5e-3,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    clean = make_toy()
    clean_types = clean.slot_types
    schedule = keep_schedule(T)

    encoder = SlotEncoder(in_dim=FEATURE_DIM, D=D)
    head = StructureHead(D=D, num_slot_types=NUM_SLOT_TYPES)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    losses: List[float] = []
    encoder.train()
    head.train()
    for step in range(steps):
        t = int(torch.randint(1, T, (1,), generator=gen).item())
        noised = corrupt(clean, t, T, schedule=schedule, generator=gen)
        feats = _features(noised, t, T)

        h_v = encoder(feats)
        _eps_hat, slot_logits = head(h_v)
        loss = head.structure_loss(slot_logits, clean_types)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # --- reverse denoising from an all-ABSENT prior ------------------------
    encoder.eval()
    head.eval()
    current = PaddedGraph(
        torch.full((clean.n_slots,), ABSENT, dtype=torch.long),
        torch.zeros(clean.n_slots, dtype=torch.float32),
    )
    transitions: List[Tuple[int, int]] = []  # (reverse_step, slot_index)
    absent_curve: List[float] = [absent_fraction(current.slot_types)]

    with torch.no_grad():
        for r_step, t in enumerate(reversed(range(T))):
            feats = _features(current, t, T)
            _eps_hat, slot_logits = head(encoder(feats))
            pred_types = slot_logits.argmax(dim=-1)

            was_absent = current.slot_types == ABSENT
            now_active = pred_types != ABSENT
            for slot in torch.nonzero(was_absent & now_active, as_tuple=False).flatten().tolist():
                transitions.append((r_step, int(slot)))

            current = PaddedGraph(pred_types, torch.zeros_like(current.values))
            absent_curve.append(absent_fraction(current.slot_types))

    return {
        "clean_types": clean_types.tolist(),
        "recovered_types": current.slot_types.tolist(),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_min": min(losses),
        "steps": steps,
        "T": T,
        "transitions": transitions,
        "absent_curve": absent_curve,
        "schedule_endpoints": (float(schedule[0]), float(schedule[-1])),
    }


def write_report(res: Dict, path: str = REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = len(res["transitions"]) > 0
    first = res["transitions"][0] if ok else None
    lines = [
        "# P3 Structure-Diffusion Pilot Report",
        "",
        "Prototype of denoising graph **structure** (adding auxiliary nodes) via the",
        "padded-slot / absorbing-D3PM formulation. Categorical forward corruption",
        "(`marc/structure/diffusion.py`) decays slot types toward `ABSENT`; the reverse",
        "head (`marc/model/structure_head.py`, Quang) is trained to repopulate them.",
        "",
        "## Setup",
        "",
        f"- Toy: {sum(1 for c in res['clean_types'] if c != ABSENT)} active slots in a pool of "
        f"{len(res['clean_types'])} (`make_toy`)",
        f"- Diffusion steps T = {res['T']}, absorbing schedule keep-prob "
        f"{res['schedule_endpoints'][0]:.3f} (t=0) -> {res['schedule_endpoints'][1]:.3f} (t=T-1)",
        f"- Training steps = {res['steps']} (<= 1000, preliminary)",
        "",
        "## Training",
        "",
        f"- Structure cross-entropy loss: {res['loss_start']:.4f} (start) -> "
        f"{res['loss_end']:.4f} (end), min {res['loss_min']:.4f}",
        "",
        "## Done-condition: a slot going ABSENT -> active",
        "",
    ]
    if ok:
        lines += [
            f"**PASS** — {len(res['transitions'])} ABSENT->active transition(s) during the "
            "reverse process from an all-ABSENT prior.",
            "",
            f"- First transition: reverse-step {first[0]}, slot #{first[1]}.",
            f"- Clean types:     `{res['clean_types']}`",
            f"- Recovered types: `{res['recovered_types']}`",
            f"- ABSENT fraction over reverse process: "
            f"{res['absent_curve'][0]:.2f} -> {res['absent_curve'][-1]:.2f}",
        ]
    else:
        lines += ["**FAIL** — no ABSENT->active transition observed."]
    lines += ["", "_Generated by `scripts/train_structure_pilot.py`._", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 structure-diffusion pilot")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=REPORT_PATH)
    args = parser.parse_args()

    res = train(steps=args.steps, seed=args.seed)
    write_report(res, args.out)

    ok = len(res["transitions"]) > 0
    print(f"loss {res['loss_start']:.4f} -> {res['loss_end']:.4f}")
    print(f"clean     : {res['clean_types']}")
    print(f"recovered : {res['recovered_types']}")
    print(f"ABSENT->active transitions: {len(res['transitions'])}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
