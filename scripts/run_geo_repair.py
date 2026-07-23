#!/usr/bin/env python3
"""Train and evaluate geometry auxiliary-construction repair (v0.4).

Population = 2D pruned distance-geometry chains the reference pipeline
(REFERENCE_SOLVER multistart + exact checker) fails as posed under two
independent restart streams. The ranker scores each candidate construction's
augmented graph; the chosen construction gets one more reference solve.

Arms (all graded end-to-end on a fresh common stream unless noted):
  ranker            top-1 of the graph-conditioned ranker         (+K_REF restarts)
  recipe_only       top-1 of the construction-recipe-only control (+K_REF)
  random            uniform construction                          (+K_REF)
  best_fixed        the single construction name with the best train work-rate
  all_cos           every cosine lift at once, no learning        (+K_REF)
  restart_control   K_REF MORE plain restarts — the matched-budget answer
  restart_plus16/32 the restart-scaling curve (unmatched, shows the plateau)
  probe             1-restart screen per construction on its own stream; an
                    accept IS a checker-accepted solution and is credited at
                    the restarts actually spent
  enumeration       try constructions in random order until one works (ceiling)

Menus have measured, possibly-multiple positive labels (the global mirror
makes sign pairs interchangeable), so training is per-candidate BCE, headline
metrics are end-to-end flips, and significance is the exact paired McNemar
test on common-stream outcomes, per house style. Trained-k and held-out-k
pools are reported separately.
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
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.eval.metrics import rate_cell, two_proportion_z
from marc.graph.semantics import build_semantic_heterodata
from marc.model.repair_ranker import GraphRepairRanker
from marc.structure.geo_repair import (
    CONSTRUCTION_FEATURE_DIM,
    GEO_REPAIR_VERSION,
    STREAM_SALT,
    ConstructionOnlyRanker,
    construction_features,
    givens_hash,
    make_dataset,
    solve_graph,
)


class GraphPlusRecipeRanker(torch.nn.Module):
    """Full model: the candidate-conditioned GNN score plus an additive head on
    the construction recipe features — the graph must carry the instance
    signal, but the model no longer has to re-derive kind/position/sign from
    topology alone."""

    def __init__(self, D: int = 96, L: int = 3):
        super().__init__()
        self.gnn = GraphRepairRanker(D=D, L=L)
        self.recipe = torch.nn.Sequential(
            torch.nn.Linear(CONSTRUCTION_FEATURE_DIM, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, graph_batch, feats):
        return self.gnn(graph_batch) + self.recipe(feats).squeeze(-1)
from marc.structure.invention_data import REFERENCE_SOLVER

K_REF = REFERENCE_SOLVER["k_refine"]


@dataclass
class Packed:
    inst: object
    graphs: list
    feats: torch.Tensor
    labels: torch.Tensor


def build_split(ks, n_per_k, seed, label_streams=3, label_restarts=None, workers=0):
    packs = []
    for inst in make_dataset(n_per_k, seed, ks=tuple(ks), label_streams=label_streams,
                             label_restarts=label_restarts, workers=workers):
        packs.append(Packed(
            inst=inst,
            graphs=[build_semantic_heterodata(c.apply(inst.graph))
                    for c in inst.constructions],
            feats=torch.stack([construction_features(c, inst.k)
                               for c in inst.constructions]),
            labels=torch.tensor(inst.worked, dtype=torch.float32),
        ))
    return packs


def _flat_batch(packs, device):
    gb = Batch.from_data_list([g for p in packs for g in p.graphs]).to(device)
    feats = torch.cat([p.feats for p in packs], dim=0).to(device)
    labels = torch.cat([p.labels for p in packs], dim=0).to(device)
    sizes = [len(p.graphs) for p in packs]
    return gb, feats, labels, sizes


@torch.no_grad()
def top1_hit_rates(full, control, packs, batch_size, device):
    """Fraction of instances whose argmax candidate is measured-working
    (label-stream metric, used for model selection only)."""
    full.eval(); control.eval()
    f_ok = c_ok = n = 0
    for start in range(0, len(packs), batch_size):
        ps = packs[start:start + batch_size]
        gb, feats, _labels, sizes = _flat_batch(ps, device)
        fs = full(gb, feats).cpu()
        cs = control(feats).cpu()
        off = 0
        for p, sz in zip(ps, sizes):
            lab = p.labels
            f_ok += bool(lab[int(fs[off:off + sz].argmax())])
            c_ok += bool(lab[int(cs[off:off + sz].argmax())])
            off += sz
            n += 1
    return f_ok / max(n, 1), c_ok / max(n, 1)


def train(full, control, train_set, val_set, *, epochs, batch_size, lr, opt_seed,
          device):
    torch.manual_seed(opt_seed)
    rng = random.Random(opt_seed)
    full.to(device); control.to(device)
    opt = torch.optim.AdamW(list(full.parameters()) + list(control.parameters()),
                            lr=lr, weight_decay=1e-5)
    bce = torch.nn.BCEWithLogitsLoss()
    best_f = best_c = -1.0
    best_full = copy.deepcopy(full.state_dict())
    best_control = copy.deepcopy(control.state_dict())
    history = []
    for epoch in range(epochs):
        rng.shuffle(train_set)
        full.train(); control.train()
        total = 0.0
        for start in range(0, len(train_set), batch_size):
            ps = train_set[start:start + batch_size]
            gb, feats, labels, _sizes = _flat_batch(ps, device)
            loss = bce(full(gb, feats), labels) + bce(control(feats), labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(full.parameters()) + list(control.parameters()), 2.0)
            opt.step()
            total += float(loss.item()) * len(ps)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            fa, ca = top1_hit_rates(full, control, val_set, batch_size, device)
            if fa > best_f:
                best_f, best_full = fa, copy.deepcopy(full.state_dict())
            if ca > best_c:
                best_c, best_control = ca, copy.deepcopy(control.state_dict())
            row = {"epoch": epoch + 1, "loss": total / max(len(train_set), 1),
                   "full_val": fa, "control_val": ca}
            history.append(row)
            print(f"epoch {epoch+1:4d}/{epochs} loss={row['loss']:.4f} "
                  f"full={fa:.3f} recipe-only={ca:.3f}", flush=True)
    full.load_state_dict(best_full)
    control.load_state_dict(best_control)
    return history, best_f, best_c


def best_fixed_name(train_set):
    """The single construction name with the highest train work-rate — the
    no-learning 'always do this' baseline."""
    wins, counts = {}, {}
    for p in train_set:
        for cons, ok in zip(p.inst.constructions, p.inst.worked):
            counts[cons.name] = counts.get(cons.name, 0) + 1
            wins[cons.name] = wins.get(cons.name, 0) + bool(ok)
    rates = {n: wins[n] / counts[n] for n in counts if counts[n] >= 5}
    return max(rates, key=rates.get) if rates else "gauge_y0_p"


def _apply_all_cos(inst):
    g = inst.graph
    for cons in inst.constructions:
        if cons.kind == "cos":
            g = cons.apply(g)
    return g


def _mcnemar(win: int, loss: int) -> float:
    d = win + loss
    if d == 0:
        return 0.5
    return sum(math.comb(d, j) for j in range(win, d + 1)) / (2.0 ** d)


@torch.no_grad()
def evaluate(full, control, packs, *, batch_size, device, opt_seed, fixed_name):
    full.eval(); control.eval()
    arms = ("ranker", "recipe_only", "random", "best_fixed", "all_cos",
            "restart_control", "restart_plus16", "restart_plus32",
            "probe", "enumeration")
    flips = {a: 0 for a in arms}
    restarts = {a: 0 for a in arms}
    wall = {a: 0.0 for a in arms}
    rows = []
    rng = random.Random(opt_seed)
    scored = []
    for start in range(0, len(packs), batch_size):
        ps = packs[start:start + batch_size]
        gb, feats, _labels, sizes = _flat_batch(ps, device)
        fs = full(gb, feats).cpu()
        cs = control(feats).cpu()
        off = 0
        for p, sz in zip(ps, sizes):
            scored.append((p, fs[off:off + sz], cs[off:off + sz]))
            off += sz

    def timed_solve(arm, graph, seed, k_restarts=None):
        t0 = time.perf_counter()
        ok = solve_graph(graph, seed=seed, k_restarts=k_restarts)
        wall[arm] += time.perf_counter() - t0
        restarts[arm] += K_REF if k_restarts is None else k_restarts
        flips[arm] += ok
        return ok

    for p, fscore, cscore in scored:
        inst = p.inst
        e2e_seed = inst.seed + 3 * STREAM_SALT   # fresh stream, common across arms
        cons = inst.constructions
        row = {"id": inst.id, "k": inst.k, "n_working": int(p.labels.sum()),
               "n_candidates": len(cons)}
        fixed_pick = next((j for j, c in enumerate(cons) if c.name == fixed_name), 0)
        picks = {
            "ranker": int(fscore.argmax()),
            "recipe_only": int(cscore.argmax()),
            "random": rng.randrange(len(cons)),
            "best_fixed": fixed_pick,
        }
        for arm, j in picks.items():
            ok = timed_solve(arm, cons[j].apply(inst.graph), e2e_seed)
            row[arm] = {"pick": cons[j].name, "flip": ok}
        row["all_cos"] = {"flip": timed_solve("all_cos", _apply_all_cos(inst), e2e_seed)}
        row["restart_control"] = {"flip": timed_solve("restart_control", inst.graph, e2e_seed)}
        row["restart_plus16"] = {"flip": timed_solve("restart_plus16", inst.graph,
                                                     e2e_seed, k_restarts=16)}
        row["restart_plus32"] = {"flip": timed_solve("restart_plus32", inst.graph,
                                                     e2e_seed, k_restarts=32)}
        # probe: 1-restart screen per construction on its OWN stream; an accept
        # is a checker-accepted solution of the augmented (hence original)
        # system, so it is credited directly at the cost actually spent
        t0 = time.perf_counter()
        probe_ok, spent = False, 0
        for j, cn in enumerate(cons):
            spent += 1
            if solve_graph(cn.apply(inst.graph),
                           seed=inst.seed + 4 * STREAM_SALT + 31 * j, k_restarts=1):
                probe_ok = True
                row["probe"] = {"pick": cn.name, "flip": True}
                break
        if not probe_ok:
            row["probe"] = {"pick": None, "flip": False}
        wall["probe"] += time.perf_counter() - t0
        restarts["probe"] += spent
        flips["probe"] += probe_ok
        # enumeration ceiling at full grade
        order = list(range(len(cons)))
        random.Random(e2e_seed + 11).shuffle(order)
        t0 = time.perf_counter()
        ok = False
        for j in order:
            restarts["enumeration"] += K_REF
            if solve_graph(cons[j].apply(inst.graph), seed=e2e_seed):
                ok = True
                break
        wall["enumeration"] += time.perf_counter() - t0
        flips["enumeration"] += ok
        row["enumeration"] = {"flip": ok}
        rows.append(row)

    def pool_stats(rs, tag):
        n = len(rs)
        out = {"n_failures": n}
        for arm in arms:
            k_flip = sum(bool(r[arm]["flip"]) for r in rs)
            out[arm] = {"flip": rate_cell(k_flip, n)}
        for baseline in ("restart_control", "random", "recipe_only",
                         "best_fixed", "all_cos", "probe"):
            win = sum(1 for r in rs if r["ranker"]["flip"] and not r[baseline]["flip"])
            lss = sum(1 for r in rs if r[baseline]["flip"] and not r["ranker"]["flip"])
            z, pval = two_proportion_z(
                sum(bool(r["ranker"]["flip"]) for r in rs), n,
                sum(bool(r[baseline]["flip"]) for r in rs), n)
            out[f"ranker_gt_{baseline}"] = {
                "paired_mcnemar": {"ranker_only": win, "baseline_only": lss,
                                   "p_one_sided_exact": _mcnemar(win, lss)},
                "z_secondary": z, "p_z_one_sided": pval,
            }
        out["pool"] = tag
        return out

    result = {
        "workable_fraction": sum(r["n_working"] for r in rows)
        / max(sum(r["n_candidates"] for r in rows), 1),
        "cost": {arm: {"restarts_per_instance": restarts[arm] / max(len(rows), 1),
                       "wall_s_per_instance": wall[arm] / max(len(rows), 1)}
                 for arm in arms},
        "rows": rows,
    }
    return result, pool_stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="geometry auxiliary-construction repair")
    ap.add_argument("--train-ks", default="10,12")
    ap.add_argument("--transfer-ks", default="14",
                    help="test-only chain lengths, reported as a separate pool")
    ap.add_argument("--n-train", type=int, default=500, help="chains generated per k")
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--D", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260722, help="data seed")
    ap.add_argument("--opt-seed", type=int, default=None,
                    help="optimization/model seed (default: data seed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/p_geo_repair/geo_repair.json")
    ap.add_argument("--ckpt", default="checkpoints/geo_repair.pt")
    ap.add_argument("--label-streams", type=int, default=3,
                    help="labels = majority vote over this many independent streams")
    ap.add_argument("--train-label-restarts", type=int, default=None,
                    help="label the TRAIN split at this per-solve budget instead of "
                         "the reference budget (1 + --label-streams 1 on the train "
                         "side = raw probe outcomes, ~12x cheaper per instance); "
                         "val/test always keep reference-budget labels")
    ap.add_argument("--train-label-streams", type=int, default=None,
                    help="override --label-streams for the TRAIN split only")
    ap.add_argument("--dataset-workers", type=int, default=0,
                    help="fan dataset generation over this many processes")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_train, args.n_val, args.n_test = 60, 30, 40
        args.epochs, args.D, args.L = 10, 48, 2
    opt_seed = args.seed if args.opt_seed is None else args.opt_seed

    train_ks = [int(x) for x in args.train_ks.split(",")]
    transfer_ks = [int(x) for x in args.transfer_ks.split(",") if x.strip()]
    test_ks = train_ks + transfer_ks
    device = torch.device(args.device)
    t0 = time.time()
    print("building hard-failure populations "
          "(generate -> 2-stream direct-solve -> keep failures -> label)", flush=True)
    train_ls = (args.label_streams if args.train_label_streams is None
                else args.train_label_streams)
    train_set = build_split(train_ks, args.n_train, args.seed, train_ls,
                            label_restarts=args.train_label_restarts,
                            workers=args.dataset_workers)
    val_set = build_split(train_ks, args.n_val, args.seed + 500000,
                          args.label_streams, workers=args.dataset_workers)
    test_set = build_split(test_ks, args.n_test, args.seed + 900000,
                           args.label_streams, workers=args.dataset_workers)
    print(f"failures: train={len(train_set)} val={len(val_set)} test={len(test_set)} "
          f"(wall {time.time()-t0:.0f}s)", flush=True)

    hashes = {name: {givens_hash(p.inst.givens) for p in split}
              for name, split in (("train", train_set), ("validation", val_set),
                                  ("test", test_set))}
    names = sorted(hashes)
    content_overlap = sum(len(hashes[a] & hashes[b])
                          for i, a in enumerate(names) for b in names[i + 1:])

    full = GraphPlusRecipeRanker(D=args.D, L=args.L)
    control = ConstructionOnlyRanker(D=64)
    history, best_f, best_c = train(
        full, control, train_set, val_set, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, opt_seed=opt_seed, device=device)
    fixed_name = best_fixed_name(train_set)
    result, pool_stats = evaluate(full, control, test_set,
                                  batch_size=args.batch_size, device=device,
                                  opt_seed=opt_seed + 42, fixed_name=fixed_name)
    trained_rows = [r for r in result["rows"] if r["k"] in train_ks]
    transfer_rows = [r for r in result["rows"] if r["k"] in transfer_ks]
    result["pools"] = {"trained": pool_stats(trained_rows, "trained")}
    if transfer_rows:
        result["pools"]["transfer"] = pool_stats(transfer_rows, "transfer")
    result["best_fixed_name"] = fixed_name

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "full_state_dict": {k: v.detach().cpu() for k, v in full.state_dict().items()},
        "control_state_dict": {k: v.detach().cpu() for k, v in control.state_dict().items()},
        "model_kwargs": {"D": args.D, "L": args.L},
        "config": vars(args),
    }, args.ckpt)
    payload = {
        "status": "ok",
        "method": "geometry auxiliary-construction repair "
                  "(derived branch pins / gauges / cosine lifts, measured labels, "
                  "2-stream hard-failure population)",
        "geo_repair_version": GEO_REPAIR_VERSION,
        "reference_solver": dict(REFERENCE_SOLVER),
        "config": {**vars(args), "opt_seed": opt_seed},
        "seed_hygiene": {
            "train_base": args.seed, "validation_base": args.seed + 500000,
            "test_base": args.seed + 900000,
            "content_hash_overlap": content_overlap,
            "purpose_streams": "STREAM_SALT-separated: fail +0/+1, label +2, "
                               "e2e +3, probe +4",
        },
        "best_validation": {"full": best_f, "control": best_c},
        "history": history,
        "result": result,
        "wall_s": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    tr = result["pools"]["trained"]
    line = " ".join(f"{a}={tr[a]['flip']['rate']:.3f}"
                    for a in ("ranker", "restart_control", "random", "best_fixed",
                              "probe", "enumeration"))
    print(f"trained-k pool: {line}", flush=True)
    if "transfer" in result["pools"]:
        tf = result["pools"]["transfer"]
        line = " ".join(f"{a}={tf[a]['flip']['rate']:.3f}"
                        for a in ("ranker", "restart_control", "random", "enumeration"))
        print(f"transfer pool (k={args.transfer_ks}): {line}", flush=True)
    print(f"wrote {out}; wall={payload['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
