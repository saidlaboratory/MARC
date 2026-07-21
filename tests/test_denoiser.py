import pytest
import torch
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter

from marc.graph.graph import FactorGraph
from marc.graph.pyg import build_heterodata
from marc.graph.schema import VariableNode, FactorNode, Edge
from marc.model.denoiser import GraphDenoiser
from marc.model.embeddings import sinusoidal_embedding


def _make_mock_graph(n_vars=2, n_facs=2):
    data = HeteroData()
    data["variable"].x = torch.randn(n_vars, 1)
    data["factor"].x = torch.zeros(n_facs, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    data["variable", "connected_to", "factor"].edge_index = edge_index
    data["variable", "connected_to", "factor"].edge_attr = edge_attr
    return data


def _make_graph(n_vars, n_facs):
    """Fully connected variable-factor graph with random attrs."""
    data = HeteroData()
    data["variable"].x = torch.randn(n_vars, 1)
    data["factor"].x = torch.randn(n_facs, 1)
    src = torch.arange(n_vars).repeat_interleave(n_facs)
    dst = torch.arange(n_facs).repeat(n_vars)
    data["variable", "connected_to", "factor"].edge_index = torch.stack([src, dst])
    data["variable", "connected_to", "factor"].edge_attr = torch.randn(n_vars * n_facs, 1)
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


def test_supports_per_graph_t_flag():
    # Pinned contract C1 — training code dispatches on this attribute.
    model = GraphDenoiser(D=16, L=1, step_dim=8)
    assert getattr(model, "supports_per_graph_t", False) is True


def test_scalar_equals_singleton_t():
    model = GraphDenoiser(D=32, L=2, step_dim=16)
    data = _make_mock_graph()
    with torch.no_grad():
        out_scalar = model(data, torch.tensor(50))
        out_singleton = model(data, torch.tensor([50]))
    assert torch.equal(out_scalar, out_singleton)


def test_per_graph_t_matches_individual_forwards():
    model = GraphDenoiser(D=32, L=2, step_dim=16)
    g1 = _make_graph(2, 2)
    g2 = _make_graph(3, 2)
    t1, t2 = 7, 63
    with torch.no_grad():
        out1 = model(g1, torch.tensor([t1]))
        out2 = model(g2, torch.tensor([t2]))
        batch = Batch.from_data_list([g1, g2])
        out_b = model(batch, torch.tensor([t1, t2]))
    assert out_b.shape == (5, 1)
    assert torch.allclose(out_b[:2], out1, atol=1e-5)
    assert torch.allclose(out_b[2:], out2, atol=1e-5)


def test_vector_t_requires_batch():
    model = GraphDenoiser(D=16, L=1, step_dim=8)
    data = _make_graph(2, 2)  # plain HeteroData, no .batch vector
    with pytest.raises(ValueError):
        model(data, torch.tensor([1, 2]))


def test_vector_t_rejects_cas_engine():
    model = GraphDenoiser(D=16, L=1, step_dim=8)
    batch = Batch.from_data_list([_make_graph(2, 2), _make_graph(3, 2)])
    with pytest.raises(ValueError):
        model(batch, torch.tensor([1, 2]), cas_engine=object())


def _const_graph():
    """x + y - 3 = 0 (const -3) and x - 5 = 0 (const -5); x touches both."""
    graph = FactorGraph(
        variables=[VariableNode("x", 0.5), VariableNode("y", -0.5)],
        factors=[FactorNode("f0", "x + y - 3"), FactorNode("f1", "x - 5")],
        edges=[Edge("x", "f0"), Edge("y", "f0"), Edge("x", "f1")],
    )
    return build_heterodata(graph)


def test_incident_const_sums_incident_factor_constants():
    data = _const_graph()
    assert torch.equal(data["factor"].x, torch.tensor([[-3.0], [-5.0]]))
    # Same gather as the denoiser forward (denoiser.py): each variable sums the
    # constant terms of its incident factors.
    src, dst = data["variable", "connected_to", "factor"].edge_index
    incident_const = scatter(data["factor"].x[dst], src, dim=0, dim_size=2, reduce="sum")
    assert torch.equal(incident_const, torch.tensor([[-8.0], [-3.0]]))


def test_output_depends_on_const_skip():
    torch.manual_seed(0)
    model = GraphDenoiser(D=32, L=2, step_dim=16)
    data = _const_graph()
    t = torch.tensor([50])
    with torch.no_grad():
        out = model(data, t)
        model.const_skip.weight.zero_()
        model.const_skip.bias.zero_()
        out_no_skip = model(data, t)
    assert not torch.allclose(out, out_no_skip)


def test_sinusoidal_embedding_vectorized():
    t = torch.tensor([1, 5, 9])
    emb = sinusoidal_embedding(t, 16)
    assert emb.shape == (3, 16)
    for i, ti in enumerate([1, 5, 9]):
        assert torch.equal(emb[i], sinusoidal_embedding(torch.tensor([ti]), 16)[0])
