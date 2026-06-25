import torch
import torch.nn as nn
import pytest
from torch_geometric.data import HeteroData

from marc.diffusion.schedule import cosine_beta_schedule
from marc.train.stage_b import sample_trajectory, compute_reward, grpo_step, train_stage_b


# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------

class StubPolicy(nn.Module):
    def __init__(self, n_vars: int = 2):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, data, t):
        x = data["variable"].x
        return self.linear(x)


def make_mock_data(n_vars: int = 2) -> HeteroData:
    d = HeteroData()
    d["variable"].x = torch.zeros(n_vars, 1)
    d["factor"].x = torch.zeros(2, 1)
    d["variable", "connected_to", "factor"].edge_index = torch.tensor(
        [[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long
    )
    return d


class MockCAS:
    def __init__(self, accept: bool = False, energy_val: float = 1.0):
        self.accept_val = accept
        self.energy_val = energy_val

    def accepts(self, x_vals, tol: float = 1e-6) -> bool:
        return self.accept_val

    def energy(self, x_vals) -> float:
        return self.energy_val

    def residuals(self, x_vals):
        return [self.energy_val ** 0.5]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sample_trajectory_shapes():
    _, alpha_bar = cosine_beta_schedule(50)
    policy = StubPolicy(n_vars=2)
    data = make_mock_data(n_vars=2)
    traj = sample_trajectory(policy, data, n_vars=2, alpha_bar=alpha_bar, steps=5)
    assert traj["x_final"].shape == (2, 1)
    assert traj["log_prob"].shape == ()  # scalar
    assert isinstance(traj["energy_trajectory"], list)


def test_compute_reward_accepted():
    traj = {
        "x_final": torch.tensor([[2.0], [1.0]]),
        "log_prob": torch.tensor(0.0),
        "energy_trajectory": [],
    }
    cas = MockCAS(accept=True, energy_val=0.0)
    r = compute_reward(traj, cas, ["x", "y"], B=10.0, use_energy_shaping=False)
    assert r == pytest.approx(10.0)


def test_compute_reward_rejected():
    traj = {
        "x_final": torch.zeros(2, 1),
        "log_prob": torch.tensor(0.0),
        "energy_trajectory": [],
    }
    cas = MockCAS(accept=False, energy_val=5.0)
    r = compute_reward(traj, cas, ["x", "y"], B=10.0, use_energy_shaping=False)
    assert r == pytest.approx(0.0)


def test_compute_reward_energy_shaping():
    traj = {
        "x_final": torch.zeros(2, 1),
        "log_prob": torch.tensor(0.0),
        "energy_trajectory": [],
    }
    cas = MockCAS(accept=False, energy_val=3.0)
    r = compute_reward(traj, cas, ["x", "y"], B=10.0, use_energy_shaping=True)
    assert r == pytest.approx(-3.0)


def test_grpo_step_returns_stats():
    _, alpha_bar = cosine_beta_schedule(50)
    policy = StubPolicy(n_vars=2)
    data = make_mock_data(n_vars=2)
    cas = MockCAS(accept=False, energy_val=2.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = grpo_step(
        policy, None, data, n_vars=2, cas_engine=cas,
        variable_ids=["x", "y"], alpha_bar=alpha_bar,
        optimizer=optimizer, N=2, steps=3,
    )
    assert "loss" in stats
    assert "mean_reward" in stats
    assert "accept_rate" in stats
    assert isinstance(stats["loss"], float)
    assert 0.0 <= stats["accept_rate"] <= 1.0


def test_grpo_step_with_ref_policy():
    _, alpha_bar = cosine_beta_schedule(50)
    policy = StubPolicy(n_vars=2)
    ref_policy = StubPolicy(n_vars=2)
    data = make_mock_data(n_vars=2)
    cas = MockCAS(accept=True, energy_val=0.0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = grpo_step(
        policy, ref_policy, data, n_vars=2, cas_engine=cas,
        variable_ids=["x", "y"], alpha_bar=alpha_bar,
        optimizer=optimizer, N=2, steps=3,
    )
    assert isinstance(stats["loss"], float)


def test_train_stage_b_smoke(tmp_path):
    _, alpha_bar = cosine_beta_schedule(50)
    policy = StubPolicy(n_vars=2)

    from marc.graph.schema import VariableNode, FactorNode, Edge
    from marc.graph.graph import FactorGraph

    graph = FactorGraph(
        variables=[VariableNode("x", 0.0), VariableNode("y", 0.0)],
        factors=[FactorNode("eq1", "x+y-3")],
        edges=[Edge("x", "eq1", 1.0), Edge("y", "eq1", 1.0)],
    )
    cas = MockCAS(accept=False, energy_val=1.0)
    trained = train_stage_b(
        policy, None,
        [(graph, {"x": 2.0, "y": 1.0}, cas)],
        alpha_bar,
        epochs=1,
        N=2,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    assert trained is policy
    assert (tmp_path / "ckpts" / "epoch_1.pt").exists()

    ckpt = torch.load(tmp_path / "ckpts" / "epoch_1.pt", weights_only=False)
    assert ckpt["epoch"] == 1
    assert "model_state_dict" in ckpt
