"""Symmetry-breaking probe: which mechanism recovers CircleLine.

Follow-up to scripts/diagnose_circleline.py, which showed the x/y nodes have
identical graph neighborhoods, so the equivariant denoiser collapses both
outputs onto the diagonal x=y and best-of-K polish solves 0/200. Canonical
targets alone cannot fix that — identical inputs give identical outputs no
matter the target — so every arm here breaks the INPUT symmetry:

  baseline   the diagnostic recipe unchanged (collapse anchor)
  index_tag  fixed per-node tag [-1, +1], target sorted ascending
  noise_tag  random tag drawn fresh per training example AND per inference
             candidate; target follows the tag ordering (larger tag -> larger
             root coordinate), so the tie-break is learnable, not leaked
  random     random-restart reference, same polish budget
  ddim       (--ddim) real trainer + multi-step solve() route

Run:  PYTHONPATH=. python3 scripts/circleline_symmetry_probe.py [--n 200] [--epochs 100] [--ddim]
Writes results/p_hard/circleline_probe.json; the headline numbers go in the
"What breaks the tie" section of paper/notes/circleline_diagnostic.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn as nn

from marc.diffusion.forward import corrupt
from marc.graph.pyg import build_heterodata
from marc.model.denoiser import GraphDenoiser

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_circleline as dc

INDEX_TAG = torch.tensor([[-1.0], [1.0]])


class TagEncoder(nn.Module):
    """Wraps the stock VariableEncoder and adds a projected per-node tag, the
    one thing that can break the x/y input tie. Kept here, not in marc/model:
    the probe must not change the model the paper evaluates."""

    def __init__(self, base, D):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(1, D)
        self.tag = None  # [n,1], set before each forward

    def forward(self, x, type_id=None, step_emb=None, incident_const=None):
        return self.base(x, type_id, step_emb, incident_const) + self.proj(self.tag)


def tagged_net(D=128, L=4):
    net = GraphDenoiser(D=D, L=L)
    net.var_encoder = TagEncoder(net.var_encoder, D)
    return net


def canon_target(sol, tag):
    """Order the root so the node with the larger tag gets the larger coordinate."""
    lo, hi = sorted(sol)
    return [lo, hi] if float(tag[0]) <= float(tag[1]) else [hi, lo]


def train_tagged(items, epochs, tag_fn, D=128, L=4):
    """dc.train_x0 with a per-node tag and the tag-consistent canonical target."""
    torch.manual_seed(0)
    net = tagged_net(D, L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), sol) for g, sol in items]
    for _ in range(epochs):
        net.train()
        for data, sol in datas:
            tag = tag_fn()
            x0 = torch.tensor([[v] for v in canon_target(sol, tag)],
                              dtype=torch.float32) / dc.SCALE
            t = torch.randint(1, dc.T + 1, (1,))
            eps = torch.randn_like(x0)
            data["variable"].x = corrupt(x0, t, eps, dc.ALPHA_BAR)
            net.var_encoder.tag = tag
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()
            opt.step()
    net.eval()
    return net


def propose_tagged(net, items, tag_fn):
    """dc.propose with the tag drawn after the same per-candidate seed."""
    out = []
    with torch.no_grad():
        for g, sol in items:
            data = build_heterodata(g)
            nv = len(sol)
            per = []
            for s in range(dc.K):
                torch.manual_seed(1000 * s + nv)
                data["variable"].x = torch.randn(nv, 1)
                net.var_encoder.tag = tag_fn()
                per.append((net(data, torch.tensor([dc.T])) * dc.SCALE).reshape(-1).tolist())
            out.append(per)
    return out


def offdiag_stats(props):
    d = [abs(p[0] - p[1]) for per in props for p in per]
    return {"mean": mean(d), "median": median(d), "max": max(d)}


def run_arm(name, items, props):
    k, n = dc.learned_rate(items, props)
    cell = dc.rate_cell(k, n)
    od = offdiag_stats(props)
    ps = dc.proposal_stats(items, props)
    print(f"{name}: solve {k}/{n} = {cell['rate']:.3f} "
          f"[{cell['ci95'][0]:.2f},{cell['ci95'][1]:.2f}]; "
          f"|px-py| mean {od['mean']:.3g} median {od['median']:.3g} max {od['max']:.3g}; "
          f"d(nearest root) {ps['mean_dist_nearest_root']:.2f}", flush=True)
    return {"solve": cell, "offdiag": od, "proposals": ps}


DDIM_CFG = """\
model: {{D: 128, L: 4, step_dim: 64, var_attn: false}}
training: {{T: 1000, epochs_A: {epochs}, epochs_B: 0, batch_size: 16, lr_A: 1.0e-3,
            lr_B: 1.0e-5, device: cpu, seed: 0, amp: false, grad_clip: 1.0,
            warmup_frac: 0.03, num_workers: 0, ema: true, ema_decay: 0.999}}
data: {{n_train: {ntrain}, n_test: 20, templates: [CircleLine], dir: {dir}/data, seed: 0}}
grpo: {{N: 2, B: 10.0, beta: 0.01, eps_clip: 0.2, steps: 5, guidance_weight: 1.0,
        problems_subset: 2, shaping_clip: 100.0}}
checkpointing: {{dir: {dir}, save_every_n_epochs: 1000}}
"""


def run_ddim(items, epochs, out_dir):
    """Train via the real Stage-A path, sample via the real multi-step solve().
    Same K candidates, same polish/accept as the other arms; raw DDIM output
    (pre-polish) is what the offdiag stats measure."""
    from marc.diffusion.solve import solve
    from marc.eval.solver import _cas_engine_for
    from marc.train import trainer as trainer_mod

    ddim_dir = out_dir / "ddim"
    ddim_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = ddim_dir / "circleline_ddim.yaml"
    cfg_path.write_text(DDIM_CFG.format(epochs=epochs, ntrain=dc.NTRAIN, dir=ddim_dir))
    trainer_mod.main(["--config", str(cfg_path), "--out-dir", str(ddim_dir),
                      "--stage", "a", "--device", "cpu"])

    ckpt = torch.load(ddim_dir / "latest.pt", map_location="cpu", weights_only=False)
    net = GraphDenoiser(**ckpt["model_kwargs"])
    net.load_state_dict(ckpt["model_state_dict"])  # raw weights, as LearnedSolver loads
    net.eval()

    props = []
    with torch.no_grad():
        for g, sol in items:
            engine = _cas_engine_for(g)
            nv = len(sol)
            per = []
            for s in range(dc.K):
                torch.manual_seed(1000 * s + nv)
                x = solve(g, net, engine, steps=40, N=1, guidance_weight=1.0)
                if x is None:  # rollout diverged to non-finite energy
                    x = torch.zeros(nv, 1)
                per.append(x.reshape(-1).tolist())
            props.append(per)
    return run_arm("ddim", items, props)


def main() -> None:
    ap = argparse.ArgumentParser(description="CircleLine symmetry-breaking probe")
    ap.add_argument("--n", type=int, default=200, help="held-out instances")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ddim", action="store_true", help="also run the trainer+DDIM arm")
    args = ap.parse_args()

    test = dc.gen(dc.TEMPLATE, args.n, seed0=100000)
    train_items = dc.gen(dc.TEMPLATE, dc.NTRAIN, seed0=0)
    out_dir = Path("results/p_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {}

    arms["baseline"] = run_arm("baseline", test,
                               dc.propose(dc.train_x0(train_items, args.epochs), test))

    net_b = train_tagged(train_items, args.epochs, lambda: INDEX_TAG)
    arms["index_tag"] = run_arm("index_tag", test,
                                propose_tagged(net_b, test, lambda: INDEX_TAG))

    net_c = train_tagged(train_items, args.epochs, lambda: torch.randn(2, 1))
    arms["noise_tag"] = run_arm("noise_tag", test,
                                propose_tagged(net_c, test, lambda: torch.randn(2, 1)))

    rk, rn = dc.random_rate(test)
    arms["random_restart"] = {"solve": dc.rate_cell(rk, rn)}
    print(f"random_restart: solve {rk}/{rn} = {rk / rn:.3f}", flush=True)

    payload = {"n_instances": args.n, "K": dc.K, "epochs": args.epochs,
               "ntrain": dc.NTRAIN, "arms": arms}
    out_path = out_dir / "circleline_probe.json"
    out_path.write_text(json.dumps(payload, indent=2))

    if args.ddim:
        arms["ddim"] = run_ddim(test, args.epochs, out_dir)
        out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
