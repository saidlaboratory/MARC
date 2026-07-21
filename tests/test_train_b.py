"""Stage-B GRPO trainer (marc/train/stage_b.py) — corrected internals.

Covers the two bug fixes:
  * differentiable log-probs replay the RECORDED per-step states
    (recompute_log_prob), so ratio == 1 before any update;
  * the ref-policy KL is computed on the SAME trajectories (cached, no ref
    rollouts) — exactly N*steps ref forwards per grpo_step.
"""

import math

import pytest
import torch
import torch.nn as nn

from marc.cas.checker import Checker
from marc.cas.engine import CASEngine
from marc.diffusion.schedule import cosine_beta_schedule
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode
from marc.graph.serialize import load_graph, save_graph
from marc.model.denoiser import GraphDenoiser
from marc.train.reward import SHAPING_CLIP, TERMINAL_B
from marc.train.rollout import recompute_log_prob, run_rollout
from marc.train.stage_b import grpo_step, train_stage_b

GRAPH_PATH = "marc/data/examples/two_equations.json"

STATS_KEYS = {"loss", "pg_loss", "kl", "mean_reward", "accept_rate", "grad_norm"}


class TinyPolicy(nn.Module):
    """Cheap denoiser stand-in: eps_hat = linear(x). Counts forward calls."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.calls = 0

    def forward(self, data, t, cas=None):
        self.calls += 1
        return self.linear(data["variable"].x)


@pytest.fixture()
def problem():
    G = load_graph(GRAPH_PATH)
    cas = CASEngine(GRAPH_PATH, "x y")
    return G, cas


# ---------------------------------------------------------------------------
# grpo_step
# ---------------------------------------------------------------------------

def test_grpo_step_returns_finite_stats(problem):
    G, cas = problem
    torch.manual_seed(0)
    policy, ref = TinyPolicy(), TinyPolicy()
    _, alpha_bar = cosine_beta_schedule(50)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    stats = grpo_step(
        policy, ref, G, cas, alpha_bar, opt,
        checker=Checker(), N=3, steps=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert set(stats) == STATS_KEYS
    for key, val in stats.items():
        assert isinstance(val, float), key
        assert math.isfinite(val), key
    assert 0.0 <= stats["accept_rate"] <= 1.0


def test_mean_reward_bounded_by_default_clip(problem):
    """Regression: clamp-bound rollouts drove raw shaping to ~ -1e8..-1e16 and
    diverged Stage-B at D512; the default shaping_clip bounds the group reward."""
    G, cas = problem
    torch.manual_seed(0)
    policy = TinyPolicy()
    _, alpha_bar = cosine_beta_schedule(50)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    stats = grpo_step(
        policy, None, G, cas, alpha_bar, opt,
        checker=Checker(), N=3, steps=4,
        generator=torch.Generator().manual_seed(0),
    )
    assert math.isfinite(stats["mean_reward"])
    assert -SHAPING_CLIP <= stats["mean_reward"] <= TERMINAL_B + SHAPING_CLIP


def test_ratio_is_one_pre_update(problem):
    """new_lp == old_lp before any optimizer step: recompute_log_prob replays
    the recorded states, so the unchanged policy reproduces the behavioral
    log-prob exactly (this is the bug the rewrite fixes)."""
    G, _ = problem
    torch.manual_seed(1)
    policy = TinyPolicy()
    traj = run_rollout(policy, G, K=5, generator=torch.Generator().manual_seed(2))
    new_lp = recompute_log_prob(policy, G, traj)
    # float32: reconstructing the action as (eps_hat + z) then subtracting
    # eps_hat again rounds; the ratio still starts at 1 to ~1e-4.
    assert torch.allclose(new_lp, traj["log_prob"], atol=1e-3)
    assert torch.exp(new_lp - traj["log_prob"]).item() == pytest.approx(1.0, abs=1e-3)


def test_ref_policy_forward_count(problem):
    """Cached ref KL: exactly N*steps ref forwards (replay only, no ref
    rollouts). The old code sampled fresh ref trajectories: ~3x N*steps."""
    G, cas = problem
    torch.manual_seed(2)
    policy, ref = TinyPolicy(), TinyPolicy()
    _, alpha_bar = cosine_beta_schedule(50)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    N, steps = 3, 4
    grpo_step(
        policy, ref, G, cas, alpha_bar, opt, N=N, steps=steps,
        generator=torch.Generator().manual_seed(0),
    )
    assert ref.calls == N * steps
    assert policy.calls == 2 * N * steps  # N sampling rollouts + N replays


def test_grad_clip_respected(problem):
    G, cas = problem
    torch.manual_seed(3)
    policy = TinyPolicy()
    _, alpha_bar = cosine_beta_schedule(50)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    clip = 1e-3
    # shaping_clip off: the untrained policy pins every rollout at the bound,
    # zeroing the group advantage — this test needs a large raw gradient.
    stats = grpo_step(
        policy, None, G, cas, alpha_bar, opt, N=4, steps=4, grad_clip=clip,
        shaping_clip=float("inf"),
        generator=torch.Generator().manual_seed(4),
    )
    # clipping actually engaged: pre-clip norm (returned stat) exceeds the cap
    assert stats["grad_norm"] > clip
    post = torch.norm(
        torch.stack([p.grad.norm() for p in policy.parameters() if p.grad is not None])
    )
    assert post <= clip + 1e-4


def test_entropy_coef_warns_and_is_noop(problem):
    G, cas = problem
    _, alpha_bar = cosine_beta_schedule(50)

    def one_step(entropy_coef):
        torch.manual_seed(5)
        policy = TinyPolicy()
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        grpo_step(
            policy, None, G, cas, alpha_bar, opt, N=2, steps=3,
            entropy_coef=entropy_coef,
            generator=torch.Generator().manual_seed(6),
        )
        return [p.detach().clone() for p in policy.parameters()]

    with pytest.warns(UserWarning, match="entropy_coef"):
        with_entropy = one_step(0.5)
    base = one_step(0.0)
    for a, b in zip(base, with_entropy):
        assert torch.allclose(a, b)

    with pytest.raises(ValueError):
        one_step(-0.1)


# ---------------------------------------------------------------------------
# train_stage_b
# ---------------------------------------------------------------------------

def test_seeded_runs_are_reproducible(tmp_path, problem):
    G, cas = problem
    _, alpha_bar = cosine_beta_schedule(50)
    problems = [(G, None, cas)]

    def run(tag):
        torch.manual_seed(123)
        policy = TinyPolicy()
        return train_stage_b(
            policy, None, problems, alpha_bar, epochs=2, N=2, lr=1e-3,
            checkpoint_dir=str(tmp_path / tag), seed=7, steps=3,
        )

    p1, p2 = run("a"), run("b")
    for a, b in zip(p1.parameters(), p2.parameters()):
        assert torch.allclose(a, b)


def test_train_stage_b_backward_compat(tmp_path, problem):
    """Exactly the call surface scripts/train_p2_checkpoints.py uses today."""
    G, cas = problem
    _, alpha_bar = cosine_beta_schedule(50)
    torch.manual_seed(0)
    policy, ref_policy = TinyPolicy(), TinyPolicy()
    problems = [(G, None, cas)]

    trained = train_stage_b(
        policy, ref_policy, problems, alpha_bar,
        epochs=1, N=4, B=10.0, beta=0.01, lr=1e-4,
        checkpoint_dir=str(tmp_path / "ckpts"), purist=False,
    )
    assert trained is policy

    ckpt_path = tmp_path / "ckpts" / "epoch_1.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert set(ckpt) == {"epoch", "model_state_dict", "optimizer_state_dict", "mean_reward"}
    assert ckpt["epoch"] == 1


# ---------------------------------------------------------------------------
# Learning signal
# ---------------------------------------------------------------------------

def test_grpo_reward_improves(tmp_path):
    """~20 seeded grpo_step calls on the 1-var problem x = 2 with a small real
    denoiser: mean reward should trend up (or the checker starts accepting)."""
    graph = FactorGraph(
        variables=[VariableNode("x", 0.0)],
        factors=[FactorNode("f0", "x-2")],
        edges=[Edge("x", "f0", 1.0)],
    )
    path = tmp_path / "one_var.json"
    save_graph(graph, str(path))
    cas = CASEngine(str(path), ["x"])  # list: a bare "x" gives a non-iterable Symbol

    torch.manual_seed(0)
    policy = GraphDenoiser(D=32, L=2, step_dim=16)
    _, alpha_bar = cosine_beta_schedule(50)
    # lr chosen so 20 GRPO steps visibly move the mean reward: an untrained
    # denoiser explodes at the cosine tail (abar ~ 0) and starts clamp-bound,
    # so the group-relative signal needs a few strong steps to show up.
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    checker = Checker()

    rewards = []
    stats = None
    for i in range(20):
        # shaping_clip off: the untrained denoiser starts clamp-bound, so the
        # default bound would pin all rewards at -shaping_clip and erase the
        # group-relative signal this test is about.
        stats = grpo_step(
            policy, None, graph, cas, alpha_bar, opt,
            checker=checker, N=4, steps=5, shaping_clip=float("inf"),
            generator=torch.Generator().manual_seed(100 + i),
        )
        rewards.append(stats["mean_reward"])

    first5 = sum(rewards[:5]) / 5
    last5 = sum(rewards[-5:]) / 5
    assert last5 > first5 or stats["accept_rate"] > 0
