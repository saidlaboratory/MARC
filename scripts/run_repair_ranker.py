#!/usr/bin/env python3
"""Train and evaluate MARC's candidate-conditioned structural repair ranker.

The experiment is deliberately matched and small-compute friendly.  The full
model and a no-problem-context control see the same menus and optimization steps;
uniform random selection is evaluated on the same instances.  Optional end-to-end
evaluation applies each selected repair, delegates values to a classical solver,
and gates the result with MARC's exact checker.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.cas.checker import Checker
from marc.eval.metrics import rate_cell, two_proportion_z
from marc.eval.runner import Problem
from marc.eval.solver import load_solver
from marc.graph.semantics import build_semantic_heterodata
from marc.model.repair_ranker import (
    CandidateOnlyRanker,
    GraphRepairRanker,
    candidate_features,
)
from marc.structure.invention_data import (
    DATA_VERSION,
    FAMILIES_BY_SOURCE,
    REFERENCE_SOLVER,
    InventionInstance,
    make_dataset,
)


@dataclass
class Packed:
    source: str
    inst: InventionInstance
    graphs: list
    features: torch.Tensor


def _families(source: str, excluded: set[str]) -> list[str]:
    return [f for f in FAMILIES_BY_SOURCE[source] if f not in excluded]


def _mask_operator_features(g):
    """Ablation: zero the operator-identity features (factor degree/has_cross/
    has_square; edge diag-quadratic/max-exponent/cross), keeping constants,
    incidence, and magnitudes — the information level of the v0.2 slot policy."""
    g["factor"].x[:, [2, 5, 6]] = 0.0
    g[("variable", "connected_to", "factor")].edge_attr[:, [1, 2, 4]] = 0.0
    return g


def build_split(sources: list[str], n_per_source: int, seed: int, K: int,
                excluded: set[str] | None = None,
                mask_operator: bool = False) -> list[Packed]:
    if len(sources) > 5:
        raise ValueError(
            "the 100000-per-source seed stride aliases into the +500000 "
            "validation offset beyond 5 sources"
        )
    excluded = excluded or set()
    out = []
    for sidx, source in enumerate(sources):
        fams = _families(source, excluded)
        if not fams:
            continue
        instances = make_dataset(
            source, n_per_source, seed + 100000 * sidx, K=K, families=fams
        )
        for inst in instances:
            out.append(Packed(
                source=source,
                inst=inst,
                graphs=[(_mask_operator_features(build_semantic_heterodata(c.apply(inst.fixed_graph)))
                         if mask_operator else
                         build_semantic_heterodata(c.apply(inst.fixed_graph)))
                        for c in inst.candidates],
                features=torch.stack([candidate_features(inst, c) for c in inst.candidates]),
            ))
    return out


def _batch(packs: list[Packed], device: torch.device):
    graph_batch = Batch.from_data_list([g for p in packs for g in p.graphs]).to(device)
    features = torch.cat([p.features for p in packs], dim=0).to(device)
    labels = torch.tensor([p.inst.gold_idx for p in packs], dtype=torch.long, device=device)
    K = len(packs[0].inst.candidates)
    return graph_batch, features, labels, K


@torch.no_grad()
def accuracies(full, control, packs: list[Packed], batch_size: int,
               device: torch.device) -> tuple[float, float]:
    full.eval()
    control.eval()
    f_ok = c_ok = n = 0
    for start in range(0, len(packs), batch_size):
        ps = packs[start:start + batch_size]
        gb, cf, labels, K = _batch(ps, device)
        fs = full(gb).reshape(len(ps), K)
        cs = control(cf).reshape(len(ps), K)
        f_ok += int((fs.argmax(1) == labels).sum().item())
        c_ok += int((cs.argmax(1) == labels).sum().item())
        n += len(ps)
    return f_ok / max(n, 1), c_ok / max(n, 1)


def train(full, control, train_set: list[Packed], val_set: list[Packed], *,
          epochs: int, batch_size: int, lr: float, seed: int,
          device: torch.device):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    full.to(device)
    control.to(device)
    opt = torch.optim.AdamW(
        list(full.parameters()) + list(control.parameters()), lr=lr, weight_decay=1e-5
    )
    best_full = copy.deepcopy(full.state_dict())
    best_control = copy.deepcopy(control.state_dict())
    best_f = best_c = -1.0
    history = []
    for epoch in range(epochs):
        rng.shuffle(train_set)
        full.train()
        control.train()
        total = 0.0
        for start in range(0, len(train_set), batch_size):
            ps = train_set[start:start + batch_size]
            gb, cf, labels, K = _batch(ps, device)
            fs = full(gb).reshape(len(ps), K)
            cs = control(cf).reshape(len(ps), K)
            loss = F.cross_entropy(fs, labels) + F.cross_entropy(cs, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(full.parameters()) + list(control.parameters()), 2.0
            )
            opt.step()
            total += float(loss.item()) * len(ps)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            fa, ca = accuracies(full, control, val_set, batch_size, device)
            if fa > best_f:
                best_f, best_full = fa, copy.deepcopy(full.state_dict())
            if ca > best_c:
                best_c, best_control = ca, copy.deepcopy(control.state_dict())
            row = {"epoch": epoch + 1, "loss": total / len(train_set),
                   "full_val": fa, "control_val": ca}
            history.append(row)
            print(f"epoch {epoch+1:4d}/{epochs} loss={row['loss']:.4f} "
                  f"full={fa:.3f} no-context={ca:.3f}", flush=True)
    full.load_state_dict(best_full)
    control.load_state_dict(best_control)
    return history, best_f, best_c


def seed_hygiene(splits: dict, sources: list[str], seed: int,
                 n_train: int, n_val: int, n_test: int) -> dict:
    """Provenance block: the REAL per-source seed ranges build_split uses
    (including the +100000*sidx stride) and a COUNTED instance-id overlap.
    The overlap count is the authoritative disjointness check; a split passed
    as None (e.g. --eval-only) is simply absent from it."""
    offsets = {"train": (0, n_train), "validation": (500000, n_val),
               "test": (900000, n_test)}
    ranges = {}
    for sidx, source in enumerate(sources):
        base = seed + 100000 * sidx
        ranges[source] = {name: [base + off, base + off + n]
                          for name, (off, n) in offsets.items()}
    ids = {name: {p.inst.id for p in split}
           for name, split in splits.items() if split}
    names = sorted(ids)
    overlap = sum(len(ids[a] & ids[b])
                  for i, a in enumerate(names) for b in names[i + 1:])
    return {"per_source_seed_ranges": ranges, "overlap_instances": overlap}


def _paired_mcnemar(rows: list[dict], arm: str, baseline: str) -> dict:
    """Exact one-sided McNemar/binomial test on paired instance outcomes."""
    win = loss = 0
    for row in rows:
        gold = row["pack"].inst.gold_idx
        a = row[arm] == gold
        b = row[baseline] == gold
        win += int(a and not b)
        loss += int(b and not a)
    discordant = win + loss
    if discordant == 0:
        p = 0.5
    else:
        p = sum(math.comb(discordant, j) for j in range(win, discordant + 1)) \
            / (2.0 ** discordant)
    return {
        "full_only_correct": win,
        "baseline_only_correct": loss,
        "discordant": discordant,
        "p_one_sided_exact": p,
    }


def _solves(pack: Packed, pick: int, seed: int) -> bool:
    graph = pack.inst.candidates[pick].apply(pack.inst.fixed_graph)
    if pack.source == "nonlinear":
        # the certificate-matched reference solver (invention_data owns the dict;
        # the same protocol certified every distractor unsolvable)
        solver = load_solver(REFERENCE_SOLVER["name"], seed=seed)
        k = REFERENCE_SOLVER["k_refine"]
    else:
        solver = load_solver("exact")  # linear menus carry the exact rank certificate
        k = 1
    problem = Problem(id=f"{pack.inst.id}_candidate{pick}", graph=graph,
                      solution=[0.0] * len(graph.variables))
    candidates = [c for c in solver.sample(problem, k) if c is not None]
    return Checker().first_accepted(graph, candidates) is not None


@torch.no_grad()
def evaluate(full, control, packs: list[Packed], *, batch_size: int,
             device: torch.device, seed: int, solve_e2e: bool) -> dict:
    full.eval()
    control.eval()
    rows = []
    rng = random.Random(seed)
    full_forward_s = 0.0
    for start in range(0, len(packs), batch_size):
        ps = packs[start:start + batch_size]
        gb, cf, _labels, K = _batch(ps, device)
        tf = time.perf_counter()
        fs = full(gb).reshape(len(ps), K).cpu()
        full_forward_s += time.perf_counter() - tf
        cs = control(cf).reshape(len(ps), K).cpu()
        for i, p in enumerate(ps):
            rows.append({
                "pack": p,
                "full": int(fs[i].argmax().item()),
                "control": int(cs[i].argmax().item()),
                "random": rng.randrange(K),
            })

    result = {}
    n = len(rows)
    result["inference_timing"] = {
        "full_forward_s_total": full_forward_s,
        "full_forward_s_per_instance": full_forward_s / n,
        "candidate_graphs_per_instance": len(packs[0].inst.candidates),
        "note": "model forward only; deterministic graph featurization is cached",
    }
    for arm in ("full", "control", "random"):
        k = sum(r[arm] == r["pack"].inst.gold_idx for r in rows)
        result[arm] = {"invention": rate_cell(k, n)}
    for arm in ("control", "random"):
        z, p = two_proportion_z(
            result["full"]["invention"]["k"], n,
            result[arm]["invention"]["k"], n,
        )
        result[f"full_gt_{arm}"] = {
            "z": z,
            "p_one_sided": p,
            "paired_mcnemar": _paired_mcnemar(rows, "full", arm),
        }

    per_family = {}
    for source, family in sorted({(r["pack"].source, r["pack"].inst.family) for r in rows}):
        rs = [r for r in rows
              if r["pack"].source == source and r["pack"].inst.family == family]
        per_family[f"{source}:{family}"] = {
            arm: rate_cell(sum(r[arm] == r["pack"].inst.gold_idx for r in rs), len(rs))
            for arm in ("full", "control", "random")
        }
    result["per_family"] = per_family

    if solve_e2e:
        solved = {arm: 0 for arm in ("full", "control", "random", "oracle", "enumeration")}
        arm_wall = {arm: 0.0 for arm in ("full", "control", "random", "oracle", "enumeration")}
        enum_calls = 0
        for i, r in enumerate(rows):
            p = r["pack"]
            # Common random numbers: every arm receives the identical restart
            # stream for this instance, so solve differences come only from the
            # selected repair.  Reinitializing the solver inside _solves makes the
            # seed literal exact rather than dependent on arm evaluation order.
            solve_seed = seed + 100003 * i
            for arm in ("full", "control", "random"):
                ts = time.perf_counter()
                solved[arm] += int(_solves(p, r[arm], solve_seed))
                arm_wall[arm] += time.perf_counter() - ts
            ts = time.perf_counter()
            solved["oracle"] += int(_solves(p, p.inst.gold_idx, solve_seed))
            arm_wall["oracle"] += time.perf_counter() - ts
            order = list(range(len(p.inst.candidates)))
            random.Random(seed + 100003 * i + 11).shuffle(order)
            for pick in order:
                enum_calls += 1
                ts = time.perf_counter()
                if _solves(p, pick, solve_seed):
                    arm_wall["enumeration"] += time.perf_counter() - ts
                    solved["enumeration"] += 1
                    break
                arm_wall["enumeration"] += time.perf_counter() - ts
        result["solve"] = {arm: rate_cell(k, n) for arm, k in solved.items()}
        result["solve"]["wall_s"] = {
            arm: {"total": wall, "per_instance": wall / n}
            for arm, wall in arm_wall.items()
        }
        result["solve"]["enumeration_cost"] = {
            "solver_calls": enum_calls,
            "calls_per_instance": enum_calls / n,
            "wall_s": arm_wall["enumeration"],
        }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="candidate-conditioned structural repair")
    ap.add_argument("--train-data", default="aux_required")
    ap.add_argument("--mask-operator-features", action="store_true",
                    help="ablation: zero operator-identity graph features")
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--exclude-family", action="append", default=[])
    ap.add_argument("--n-train", type=int, default=500, help="instances per source")
    ap.add_argument("--n-val", type=int, default=100, help="instances per source")
    ap.add_argument("--n-test", type=int, default=500, help="instances per source")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--D", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--solve-e2e", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --ckpt and skip train/validation generation")
    ap.add_argument("--out", default="results/p_repair/repair_eval.json")
    ap.add_argument("--ckpt", default="checkpoints/repair_ranker.pt")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_train, args.n_val, args.n_test = 64, 24, 48
        args.epochs, args.D, args.L = 20, 48, 2

    train_sources = [x.strip() for x in args.train_data.split(",") if x.strip()]
    eval_sources = [x.strip() for x in (args.eval_data or args.train_data).split(",") if x.strip()]
    bad = [s for s in train_sources + eval_sources if s not in FAMILIES_BY_SOURCE]
    if bad:
        ap.error(f"unknown data sources: {bad}")
    excluded = set(args.exclude_family)
    device = torch.device(args.device)
    t0 = time.time()
    print("building disjoint train/validation/test menus", flush=True)
    train_set = val_set = None
    ckpt_payload = None
    if args.eval_only:
        ckpt_payload = torch.load(args.ckpt, map_location="cpu")
        saved = ckpt_payload["model_kwargs"]
        args.D, args.L = int(saved["D"]), int(saved["L"])
    else:
        train_set = build_split(train_sources, args.n_train, args.seed, args.K, excluded,
                                mask_operator=args.mask_operator_features)
        val_set = build_split(train_sources, args.n_val, args.seed + 500000, args.K, excluded,
                              mask_operator=args.mask_operator_features)
    test_set = build_split(eval_sources, args.n_test, args.seed + 900000, args.K,
                           mask_operator=args.mask_operator_features)
    full = GraphRepairRanker(D=args.D, L=args.L)
    control = CandidateOnlyRanker(D=args.D)
    if args.eval_only:
        full.load_state_dict(ckpt_payload["full_state_dict"])
        control.load_state_dict(ckpt_payload["control_state_dict"])
        history = []
        best_f = best_c = None
        full.to(device)
        control.to(device)
    else:
        history, best_f, best_c = train(
            full, control, train_set, val_set, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=device,
        )
    result = evaluate(
        full, control, test_set, batch_size=args.batch_size, device=device,
        seed=args.seed + 42, solve_e2e=args.solve_e2e,
    )

    if not args.eval_only:
        Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "full_state_dict": {k: v.detach().cpu() for k, v in full.state_dict().items()},
            "control_state_dict": {k: v.detach().cpu() for k, v in control.state_dict().items()},
            "model_kwargs": {"D": args.D, "L": args.L},
            "config": vars(args),
        }, args.ckpt)
    payload = {
        "status": "ok",
        "method": "operator-aware candidate-conditioned graph repair ranking",
        "data_version": DATA_VERSION,
        "certificates": {
            kind: sum(p.inst.certificate == kind for p in test_set)
            for kind in sorted({p.inst.certificate for p in test_set})
        },
        "config": vars(args),
        "seed_hygiene": seed_hygiene(
            {"train": train_set, "validation": val_set, "test": test_set},
            list(dict.fromkeys(train_sources + eval_sources)),
            args.seed, args.n_train, args.n_val, args.n_test,
        ),
        "best_validation": {"full": best_f, "control": best_c},
        "history": history,
        "result": result,
        "wall_s": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"test full={result['full']['invention']['rate']:.3f} "
          f"no-context={result['control']['invention']['rate']:.3f} "
          f"random={result['random']['invention']['rate']:.3f}", flush=True)
    print(f"wrote {out}; wall={payload['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
