import pytest
import torch
from torch_geometric.data import HeteroData, Batch

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


def test_var_attn_off_matches_old_ctor():
    # Flag off must be a no-op: same seed, same params, same outputs as before
    # the kwarg existed (no extra modules are created, so RNG streams align).
    torch.manual_seed(0)
    old = GraphDenoiser(D=32, L=2, step_dim=16)
    torch.manual_seed(0)
    new = GraphDenoiser(D=32, L=2, step_dim=16, var_attn=False)
    data = _make_mock_graph()
    t = torch.tensor([50])
    with torch.no_grad():
        assert torch.equal(old(data, t), new(data, t))


def test_var_attn_forward_backward():
    model = GraphDenoiser(D=32, L=2, step_dim=16, var_attn=True)
    data = _make_mock_graph()
    eps_hat = model(data, torch.tensor([50]))
    assert eps_hat.shape == (2, 1)
    eps_hat.sum().backward()
    assert model.attn.in_proj_weight.grad is not None


def test_var_attn_no_cross_graph_leakage():
    model = GraphDenoiser(D=32, L=2, step_dim=16, var_attn=True)
    g1 = _make_graph(2, 2)
    g2 = _make_graph(3, 2)
    g2b = g2.clone()
    g2b["variable"].x = g2b["variable"].x + 1.0
    t = torch.tensor([7, 63])
    with torch.no_grad():
        out = model(Batch.from_data_list([g1, g2]), t)
        out_pert = model(Batch.from_data_list([g1, g2b]), t)
    assert torch.allclose(out[:2], out_pert[:2], atol=1e-6)
    assert not torch.allclose(out[2:], out_pert[2:])


def test_var_attn_checkpoint_compat():
    # State dicts without attention keys must load into var_attn=False strictly
    # and into var_attn=True via strict=False (only the attn params missing).
    state = GraphDenoiser(D=32, L=2, step_dim=16).state_dict()
    GraphDenoiser(D=32, L=2, step_dim=16, var_attn=False).load_state_dict(state)
    result = GraphDenoiser(D=32, L=2, step_dim=16, var_attn=True).load_state_dict(
        state, strict=False
    )
    assert not result.unexpected_keys
    assert all(k.startswith(("attn.", "attn_norm.")) for k in result.missing_keys)


def test_sinusoidal_embedding_vectorized():
    t = torch.tensor([1, 5, 9])
    emb = sinusoidal_embedding(t, 16)
    assert emb.shape == (3, 16)
    for i, ti in enumerate([1, 5, 9]):
        assert torch.equal(emb[i], sinusoidal_embedding(torch.tensor([ti]), 16)[0])
