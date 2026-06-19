import torch
from torch_geometric.data import HeteroData

from marc.model.denoiser import GraphDenoiser


def _make_mock_graph(n_vars=2, n_facs=2):
    data = HeteroData()
    data["variable"].x = torch.randn(n_vars, 1)
    data["factor"].x = torch.zeros(n_facs, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    data["variable", "connected_to", "factor"].edge_index = edge_index
    data["variable", "connected_to", "factor"].edge_attr = edge_attr
    return data


def test_denoiser_output_shape():
    model = GraphDenoiser(D=32, L=2, step_dim=16)
    data = _make_mock_graph()
    t = torch.tensor([50])
    eps_hat = model(data, t)
    assert eps_hat.shape == (2, 1), f"Expected (2,1), got {eps_hat.shape}"


def test_denoiser_backward():
    model = GraphDenoiser(D=32, L=2, step_dim=16)
    data = _make_mock_graph()
    t = torch.tensor([100])
    eps_hat = model(data, t)
    loss = eps_hat.sum()
    loss.backward()
    # Check at least one param has gradient
    for p in model.parameters():
        if p.grad is not None:
            return
    raise AssertionError("No gradients computed")


def test_denoiser_no_cas():
    # Without cas_engine, residuals should default to zeros — no crash
    model = GraphDenoiser(D=16, L=1, step_dim=8)
    data = _make_mock_graph()
    eps_hat = model(data, torch.tensor([5]))
    assert not torch.isnan(eps_hat).any()
