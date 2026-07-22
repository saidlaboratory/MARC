"""Train the U5 structure-invention policy (menu-based, absorbing D3PM).

Per step (one instance — the graphs are tiny; CPU trains in minutes):
    t ~ U[1, T);  noised = corrupt(to_padded(inst), t, T)   # existing forward process
    loss = StructureHead.structure_loss(slot_logits, clean_types)
           + 0.1 * StructureHead.value_loss(value_pred, clean_values, clean_types)
           [+ rl_weight * REINFORCE on the menu categorical with solve reward]

Validation every 10 epochs: single-shot invention accuracy on 50 held-out-constant
instances (seed offset 500000); the best checkpoint is kept.

Usage (CLI contract C4 — the overnight harness calls this exactly):
    python3 scripts/train_structure_policy.py --out checkpoints/structure_policy.pt \
        --epochs 200 --device auto [--data toys|aux_required] [--n-train 500] [--K 4] \
        [--T 20] [--D 64] [--L 2] [--seed 0] [--exclude-family X] \
        [--no-hard-negatives] [--rl-weight W] [--filter-unsolvable] \
        [--ablate-context] [--smoke]
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marc.cas.checker import Checker
from marc.eval.runner import Problem
from marc.eval.solver import load_solver
from marc.structure import invention_data
from marc.structure.diffusion import corrupt, keep_schedule
from marc.structure.invention_data import FAMILIES, make_dataset, to_padded
from marc.structure.policy import StructurePolicy, chosen_candidate, reverse_sample
from marc.structure.schema import ABSENT, PaddedGraph, SlotType

# seed-space contract v1 — must match scripts/run_invention_eval.py
VAL_SEED_OFFSET = 500000
VAL_SIZE = 50
TEST_SEED_MIN = 900000

# reference solver — owned by invention_data, shared with run_invention_eval and
# gold/distractor certification, so training reward and eval grading cannot diverge
REFERENCE_SOLVER = invention_data.REFERENCE_SOLVER

RL_GROUP = 4  # instances per REINFORCE step


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


def menu_scores(slot_logits: torch.Tensor, K: int) -> torch.Tensor:
    """Menu logits [K+1] from the single-shot readout at t=T-1 on all-ABSENT input:
    candidate j scores logits[2j, VARIABLE] + logits[2j+1, FACTOR]; the last entry
    scores "none" as the log-probability that every slot reads ABSENT."""
    var_t, fac_t = int(SlotType.VARIABLE), int(SlotType.FACTOR)
    cand = torch.stack(
        [slot_logits[2 * j, var_t] + slot_logits[2 * j + 1, fac_t] for j in range(K)]
    )
    none = torch.log_softmax(slot_logits, dim=-1)[:, ABSENT].sum().unsqueeze(0)
    return torch.cat([cand, none])


def rl_loss_from_scores(scores_list, picks, rewards: torch.Tensor,
                        weight: float) -> torch.Tensor:
    """REINFORCE with a group-relative baseline:
    -weight * mean_i((r_i - mean(r)) * log softmax(scores_i)[pick_i]).
    All-equal rewards give zero advantage, hence exactly zero loss."""
    logps = torch.stack(
        [torch.log_softmax(s, dim=-1)[p] for s, p in zip(scores_list, picks)]
    )
    rewards = rewards.to(logps.device)
    adv = rewards - rewards.mean()
    return -weight * (adv * logps).mean()


def candidate_solves(inst, pick, solver, checker, cache: dict) -> bool:
    """Best-of-k_refine REFERENCE_SOLVER + Checker accept, cached per
    (instance, pick) — rewards are deterministic, so later epochs are nearly free.

    ``pick=None`` is the fixed graph, which build_menu certifies inconsistent:
    picking "none" can never solve, so return False (RL reward 0) without solving.
    """
    if pick is None:
        return False
    key = (inst.id, pick)
    if key not in cache:
        graph = inst.candidates[pick].apply(inst.fixed_graph)
        prob = Problem(
            id=f"{inst.id}_pick{pick}",
            graph=graph,
            solution=[0.0] * len(graph.variables),  # unused by refine
        )
        cands = [c for c in solver.sample(prob, REFERENCE_SOLVER["k_refine"])
                 if c is not None]
        cache[key] = checker.first_accepted(graph, cands) is not None
    return cache[key]


def build_model_kwargs(args) -> dict:
    """Model kwargs for StructurePolicy; guards the P1 --ablate-context contract."""
    kwargs = {"D": args.D, "L": args.L, "K": args.K}
    if args.ablate_context:
        if "ablate_context" not in inspect.signature(StructurePolicy.__init__).parameters:
            raise SystemExit(
                "--ablate-context requires marc.structure.policy with ablate_context "
                "support (unit W3); this StructurePolicy does not accept it"
            )
        # written ONLY when set — keeps older ckpts loadable by pre-W3 code
        kwargs["ablate_context"] = True
    return kwargs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the U5 structure-invention policy")
    ap.add_argument("--out", required=True, help="checkpoint output path")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="auto", help="'auto' = cuda if available else cpu")
    ap.add_argument("--data", default="toys",
                    choices=getattr(invention_data, "SOURCES", ("toys", "aux_required")))
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--rl-weight", type=float, default=0.0,
                    help="weight of the solve-reward REINFORCE term (0 = off)")
    ap.add_argument("--filter-unsolvable", action="store_true",
                    help="drop train instances whose gold candidate does not solve "
                         "at the reference solver literal")
    ap.add_argument("--ablate-context", action="store_true",
                    help="train without fixed-graph context (needs W3 StructurePolicy)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: D=32, n-train=48, epochs=30, T=10 (<2 min CPU)")
    args = ap.parse_args(argv)

    if args.smoke:
        args.D, args.n_train, args.epochs, args.T = 32, 48, 30, 10

    # seed-space contract v1: training must never touch test seeds.
    if args.seed + VAL_SEED_OFFSET + VAL_SIZE > TEST_SEED_MIN:
        raise SystemExit(
            f"seed-space violation: val seeds [{args.seed + VAL_SEED_OFFSET}, "
            f"{args.seed + VAL_SEED_OFFSET + VAL_SIZE}) cross TEST_SEED_MIN="
            f"{TEST_SEED_MIN}; lower --seed"
        )
    if args.n_train >= VAL_SEED_OFFSET:
        raise SystemExit(
            f"seed-space violation: --n-train {args.n_train} >= VAL_SEED_OFFSET="
            f"{VAL_SEED_OFFSET}; train seeds would cross into the val range"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # families default to the vocabulary of the chosen data source (aux_required uses
    # offset/coupled/shared, not the toy names) — mirrors run_invention_eval's resolution.
    families = list(invention_data.FAMILIES_BY_SOURCE.get(args.data, FAMILIES))
    if args.exclude_family:
        families = [f for f in families if f != args.exclude_family]
        if not families:
            raise SystemExit(f"--exclude-family {args.exclude_family!r} removes every family")
    hard = not args.no_hard_negatives

    train_set = make_dataset(args.data, args.n_train, args.seed, K=args.K,
                             families=families, hard_negatives=hard)
    val_set = make_dataset(args.data, VAL_SIZE, args.seed + VAL_SEED_OFFSET, K=args.K,
                           families=families, hard_negatives=hard)

    solver = checker = None
    solve_cache: dict = {}
    if args.rl_weight > 0 or args.filter_unsolvable:
        solver = load_solver(REFERENCE_SOLVER["name"], seed=args.seed)
        checker = Checker()

    def solves(inst, pick):
        # module-level lookup so tests can monkeypatch candidate_solves
        return candidate_solves(inst, pick, solver, checker, solve_cache)

    filtered = None
    if args.filter_unsolvable:
        checked = len(train_set)
        train_set = [inst for inst in train_set if solves(inst, inst.gold_idx)]
        filtered = {"checked": checked, "dropped": checked - len(train_set)}
        print(f"--filter-unsolvable: dropped {filtered['dropped']}/{checked} instances "
              f"whose gold candidate does not solve at the reference literal")
        if not train_set:
            raise SystemExit("--filter-unsolvable dropped every training instance")

    torch.manual_seed(args.seed)
    model_kwargs = build_model_kwargs(args)
    policy = StructurePolicy(**model_kwargs).to(device)
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
            if args.rl_weight > 0:
                # ponytail: one RL_GROUP-instance REINFORCE group per CE step;
                # solve rewards are cached, so the cost is ~RL_GROUP extra forwards.
                gidx = torch.randint(len(train_set), (RL_GROUP,), generator=gen).tolist()
                scores_list, picks, rewards = [], [], []
                for gi in gidx:
                    ginst = train_set[gi]
                    Kg = len(ginst.candidates)
                    prior = PaddedGraph(
                        torch.full((2 * Kg,), ABSENT, dtype=torch.long),
                        torch.zeros(2 * Kg, dtype=torch.float32),
                    )
                    _vp, glogits = policy(ginst, prior, args.T - 1, args.T)
                    s = menu_scores(glogits, Kg)
                    pick = int(torch.multinomial(
                        torch.softmax(s.detach().cpu(), dim=-1), 1, generator=gen
                    ).item())
                    choice = None if pick == Kg else pick  # "none" -> fixed graph, reward 0
                    rewards.append(1.0 if solves(ginst, choice) else 0.0)
                    scores_list.append(s)
                    picks.append(pick)
                loss = loss + rl_loss_from_scores(
                    scores_list, picks, torch.tensor(rewards), args.rl_weight
                )
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
    # seed-space contract v1 — recorded so any eval can detect train/test seed
    # overlap from the checkpoint alone.
    train_config["seed_space_version"] = 1
    train_config["train_seed_range"] = [args.seed, args.seed + args.n_train]
    train_config["val_seed_range"] = [args.seed + VAL_SEED_OFFSET,
                                      args.seed + VAL_SEED_OFFSET + VAL_SIZE]
    train_config["test_seed_min"] = TEST_SEED_MIN
    train_config["data_version"] = getattr(invention_data, "DATA_VERSION", 1)
    if filtered is not None:
        train_config["filtered"] = filtered
    torch.save(
        {
            "model_state_dict": best_state,
            "model_kwargs": model_kwargs,
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
