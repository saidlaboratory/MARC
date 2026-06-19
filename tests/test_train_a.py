import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Batch
from marc.diffusion.schedule import cosine_beta_schedule
from marc.train.stage_a import train_step_A, train_stage_a


# Minimal stub denoiser for testing — does NOT depend on marc.model
class StubDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, data, t):
        return self.linear(data["variable"].x)


def make_mock_batch(n_vars=2, batch_size=2):
    """Return a mock (Batch, solutions) pair."""
    data_list = []
    solutions = []
    for _ in range(batch_size):
        d = HeteroData()
        d["variable"].x = torch.randn(n_vars, 1)
        d["factor"].x = torch.zeros(2, 1)
        d["variable", "connected_to", "factor"].edge_index = torch.tensor(
            [[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long
        )
        data_list.append(d)
        solutions.append(torch.randn(n_vars, 1))
    batched = Batch.from_data_list(data_list)
    return batched, solutions


def test_train_step_returns_loss():
    _, alpha_bar = cosine_beta_schedule(100)
    model = StubDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_mock_batch(n_vars=2, batch_size=2)
    loss = train_step_A(model, batch, alpha_bar, T=100, optimizer=optimizer)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_train_step_updates_weights():
    _, alpha_bar = cosine_beta_schedule(100)
    model = StubDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.linear.weight.clone().detach()
    batch = make_mock_batch()
    train_step_A(model, batch, alpha_bar, T=100, optimizer=optimizer)
    after = model.linear.weight.clone().detach()
    assert not torch.allclose(before, after), "Weights should change after training step"


def test_train_stage_a_smoke(tmp_path):
    _, alpha_bar = cosine_beta_schedule(100)
    model = StubDenoiser()
    dataset = [make_mock_batch(n_vars=2, batch_size=1) for _ in range(2)]
    loader = DataLoader(dataset, batch_size=None, collate_fn=lambda x: x)
    trained = train_stage_a(
        model,
        loader,
        alpha_bar,
        T=100,
        epochs=2,
        checkpoint_dir=str(tmp_path / "ckpts"),
        lr=1e-3,
    )
    assert trained is model
    assert (tmp_path / "ckpts" / "epoch_1.pt").exists()
    assert (tmp_path / "ckpts" / "epoch_2.pt").exists()


def test_checkpoint_loadable(tmp_path):
    _, alpha_bar = cosine_beta_schedule(100)
    model = StubDenoiser()
    dataset = [make_mock_batch()]
    loader = DataLoader(dataset, batch_size=None, collate_fn=lambda x: x)
    train_stage_a(
        model,
        loader,
        alpha_bar,
        T=100,
        epochs=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    ckpt = torch.load(str(tmp_path / "ckpts" / "epoch_1.pt"))
    assert "model_state_dict" in ckpt
    assert "loss" in ckpt
