"""Train the U5 structure-invention policy (menu-based, absorbing D3PM).

Per step (one instance — the graphs are tiny; CPU trains in minutes):
    t ~ U[1, T);  noised = corrupt(to_padded(inst), t, T)   # existing forward process
    loss = StructureHead.structure_loss(slot_logits, clean_types)
           + 0.1 * StructureHead.value_loss(value_pred, clean_values, clean_types)

Validation every 10 epochs: single-shot invention accuracy on 50 held-out-constant
instances (seed offset 500000); the best checkpoint is kept.

Usage (CLI contract C4 — the overnight harness calls this exactly):
    python3 scripts/train_structure_policy.py --out checkpoints/structure_policy.pt \
        --epochs 200 --device auto [--data toys|aux_required] [--n-train 500] [--K 4] \
        [--T 20] [--D 64] [--L 2] [--seed 0] [--exclude-family X] \
        [--no-hard-negatives] [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marc.structure.diffusion import corrupt, keep_schedule
from marc.structure.invention_data import FAMILIES, make_dataset, to_padded
from marc.structure.policy import StructurePolicy, chosen_candidate, reverse_sample

VAL_SEED_OFFSET = 500000
VAL_SIZE = 50


def invention_accuracy(policy, instances, T: int, *, single_shot: bool = True,
                       generator: torch.Generator | None = None) -> float:
    """Fraction of instances where the sampled structure picks the gold candidate."""
    was_training = policy.training
    policy.eval()
    hits = 0
    for inst in instances:
        final, logits = reverse_sample(
            policy, inst, T=T, generator=generator, single_shot=single_shot
        )
        if chosen_candidate(final, logits, len(inst.candidates)) == inst.gold_idx:
            hits += 1
    if was_training:
        policy.train()
    return hits / max(len(instances), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the U5 structure-invention policy")
    ap.add_argument("--out", required=True, help="checkpoint output path")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="auto", help="'auto' = cuda if available else cpu")
    ap.add_argument("--data", choices=("toys", "aux_required"), default="toys")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: D=32, n-train=48, epochs=30, T=10 (<2 min CPU)")
    args = ap.parse_args()

    if args.smoke:
        args.D, args.n_train, args.epochs, args.T = 32, 48, 30, 10

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    families = list(FAMILIES)
    if args.exclude_family:
        families = [f for f in families if f != args.exclude_family]
        if not families:
            raise SystemExit(f"--exclude-family {args.exclude_family!r} removes every family")
    hard = not args.no_hard_negatives

    train_set = make_dataset(args.data, args.n_train, args.seed, K=args.K,
                             families=families, hard_negatives=hard)
    val_set = make_dataset(args.data, VAL_SIZE, args.seed + VAL_SEED_OFFSET, K=args.K,
                           families=families, hard_negatives=hard)

    torch.manual_seed(args.seed)
    policy = StructurePolicy(D=args.D, L=args.L, K=args.K).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(args.seed)
    schedule = keep_schedule(args.T)

    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    t0 = time.time()
    for epoch in range(args.epochs):
        policy.train()
        total = 0.0
        for idx in torch.randperm(len(train_set), generator=gen).tolist():
            inst = train_set[idx]
            clean = to_padded(inst)
            t = int(torch.randint(1, args.T, (1,), generator=gen).item())
            noised = corrupt(clean, t, args.T, schedule=schedule, generator=gen)
            value_pred, slot_logits = policy(inst, noised, t, args.T)
            tgt_types = clean.slot_types.to(slot_logits.device)
            tgt_vals = clean.values.unsqueeze(-1).to(value_pred.device)
            loss = policy.head.structure_loss(slot_logits, tgt_types) \
                + 0.1 * policy.head.value_loss(value_pred, tgt_vals, tgt_types)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            acc = invention_accuracy(policy, val_set, args.T, single_shot=True)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            print(f"epoch {epoch + 1:4d}/{args.epochs}  loss {total / len(train_set):.4f}  "
                  f"val single-shot invention acc {acc:.3f}  best {best_acc:.3f}  "
                  f"[{time.time() - t0:.0f}s]")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    train_config = dict(vars(args))
    train_config["families"] = families
    train_config["hard_negatives"] = hard
    torch.save(
        {
            "model_state_dict": best_state,
            "model_kwargs": {"D": args.D, "L": args.L, "K": args.K},
            "train_config": train_config,
        },
        args.out,
    )
    print(f"checkpoint -> {args.out}  (best val single-shot invention acc {best_acc:.3f})")

    if args.smoke:
        policy.load_state_dict(best_state)
        policy.to(device)
        train_acc = invention_accuracy(policy, train_set, args.T, single_shot=True)
        print(f"[smoke] final single-shot invention acc  train {train_acc:.3f}  val {best_acc:.3f}")


if __name__ == "__main__":
    main()
