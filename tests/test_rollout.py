"""Rollout MDP (marc/train/rollout.py): trajectory shape, energy logging, accept.

Uses a trivial mock policy (returns zeros) so the test exercises run_rollout's
plumbing — schedule, DDIM stepping, log-prob accumulation, energy/accept
recording — rather than a trained model.
"""

import torch
import torch.nn as nn

from marc.cas.engine import CASEngine
from marc.cas.checker import Checker
from marc.diffusion.schedule import cosine_beta_schedule
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.graph.serialize import load_graph
from marc.model.denoiser import GraphDenoiser
from marc.train.rollout import _cached_alpha_bar, recompute_log_prob, run_rollout

GRAPH_PATH = "marc/data/examples/two_equations.json"


class ZeroPolicy(nn.Module):
    """A stand-in denoiser: eps_hat == 0 for every node."""

    def forward(self, data, t, cas=None):
        n = data["variable"].x.shape[0]
        return torch.zeros(n, 1)


def test_rollout_trajectory_shapes():
    G = load_graph(GRAPH_PATH)
    n = len(G.variables)
    K = 5
    gen = torch.Generator().manual_seed(0)
    traj = run_rollout(ZeroPolicy(), G, K=K, generator=gen)

    for key in ("x_final", "x_values", "log_prob", "eps_hats", "raw_noises",
                "states", "step_indices"):
        assert key in traj, f"missing key {key}"

    assert traj["x_final"].shape == (n, 1)
    assert len(traj["x_values"]) == n
    assert len(traj["eps_hats"]) == K
    assert len(traj["states"]) == K
    assert len(traj["step_indices"]) == K
    # log_prob is a finite scalar
    assert traj["log_prob"].shape == ()
    assert torch.isfinite(traj["log_prob"])


def test_rollout_records_energy_and_accept():
    G = load_graph(GRAPH_PATH)
    cas = CASEngine(GRAPH_PATH, "x y")
    checker = Checker()
    gen = torch.Generator().manual_seed(1)
    traj = run_rollout(ZeroPolicy(), G, K=4, cas=cas, checker=checker, generator=gen)

    # one energy reading for the initial state plus one per step
    assert len(traj["energy_trajectory"]) == 4 + 1
    assert all(e >= 0 for e in traj["energy_trajectory"])
    assert isinstance(traj["accepted"], bool)


def test_rollout_is_reproducible_with_seed():
    G = load_graph(GRAPH_PATH)
    a = run_rollout(ZeroPolicy(), G, K=3, generator=torch.Generator().manual_seed(7))
    b = run_rollout(ZeroPolicy(), G, K=3, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a["x_final"], b["x_final"])


def test_recompute_log_prob_matches_recorded():
    """Identity check: with the policy unchanged and the recorded states replayed,
    the differentiable log-prob equals the one recorded at sampling time
    (eps_hat_new == eps_hat_old, so the residual is exactly the sampled z)."""
    torch.manual_seed(0)
    policy = GraphDenoiser(D=32, L=2, step_dim=16)
    G = FactorGraph(
        variables=[VariableNode("x", 0.0), VariableNode("y", 0.0)],
        factors=[FactorNode("eq1", "x+y-3")],
        edges=[Edge("x", "eq1", 1.0), Edge("y", "eq1", 1.0)],
    )
    traj = run_rollout(policy, G, K=5, generator=torch.Generator().manual_seed(11))
    lp = recompute_log_prob(policy, G, traj)
    assert lp.requires_grad
    assert torch.allclose(lp, traj["log_prob"], atol=1e-4)


def test_schedule_cache_returns_same_tensor():
    a = _cached_alpha_bar(123, "cpu")
    b = _cached_alpha_bar(123, "cpu")
    assert a is b
    assert torch.allclose(a, cosine_beta_schedule(123)[1])
