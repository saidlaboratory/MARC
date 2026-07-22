"""Self-trained toy x0 proposer shared by the counting eval scripts.

Hoisted verbatim from scripts/run_hard_eval.py; the per-script differences
(solution scale, torch seed, the crossfamily epoch shuffle) are parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from marc.diffusion.forward import corrupt
from marc.diffusion.schedule import cosine_beta_schedule
from marc.graph.pyg import build_heterodata
from marc.model.denoiser import GraphDenoiser

T = 1000
_, ALPHA_BAR = cosine_beta_schedule(T)


def gen(template, count, seed0):
    out = []
    for i in range(count):
        g, sol = template.generate(seed=seed0 + i)
        out.append((g, [float(v) for v in sol.values()]))
    return out


def train_x0(items, epochs, D=128, L=4, seed=0, scale=5.0, shuffle=False):
    torch.manual_seed(seed)
    net = GraphDenoiser(D=D, L=L)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    datas = [(build_heterodata(g), torch.tensor([[v] for v in sol], dtype=torch.float32) / scale)
             for g, sol in items]
    for _ in range(epochs):
        net.train()
        if shuffle:
            # ponytail: no-op shuffle kept, randperm advances global torch rng that later draws consume
            torch.randperm(len(datas))
        for data, x0 in datas:
            t = torch.randint(1, T + 1, (1,))
            eps = torch.randn_like(x0)
            data["variable"].x = corrupt(x0, t, eps, ALPHA_BAR)
            opt.zero_grad()
            nn.functional.mse_loss(net(data, t), x0).backward()
            opt.step()
    net.eval()
    return net
