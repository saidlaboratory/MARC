"""Embedding building blocks (marc/model/embeddings.py): shapes + gradient flow."""

import torch

from marc.model.embeddings import (
    FactorEncoder,
    VariableEncoder,
    sinusoidal_embedding,
)


def test_sinusoidal_embedding_shape_and_finiteness():
    t = torch.tensor([0, 1, 500, 999])
    emb = sinusoidal_embedding(t, dim=16)
    assert emb.shape == (4, 16)
    assert torch.isfinite(emb).all()
    # distinct timesteps -> distinct embeddings
    assert not torch.allclose(emb[0], emb[1])


def test_sinusoidal_embedding_requires_even_dim():
    try:
        sinusoidal_embedding(torch.tensor([1]), dim=15)
    except AssertionError:
        return
    raise AssertionError("odd dim should have raised")


def test_variable_encoder_shape_and_type_conditioning():
    enc = VariableEncoder(D=8, step_dim=4)
    x = torch.randn(5, 1)
    h = enc(x)
    assert h.shape == (5, 8)
    # supplying a type id shifts the representation
    type_id = torch.tensor([0, 1, 2, 3, 0])
    h_typed = enc(x, type_id)
    assert h_typed.shape == (5, 8)
    assert not torch.allclose(h, h_typed)
    # timestep conditioning also shifts it
    step_emb = torch.randn(5, 4)
    h_t = enc(x, type_id, step_emb)
    assert h_t.shape == (5, 8)
    assert not torch.allclose(h_typed, h_t)


def test_factor_encoder_shape_and_backward():
    D, step_dim = 8, 4
    enc = FactorEncoder(D=D, step_dim=step_dim)
    type_id = torch.tensor([0, 1, 2])
    residual = torch.randn(3, 1, requires_grad=True)
    step_emb = torch.randn(3, step_dim)
    const = torch.randn(3, 1)
    out = enc(type_id, residual, step_emb, const)
    assert out.shape == (3, D)
    out.sum().backward()
    assert residual.grad is not None
    # const is optional (defaults to zeros)
    assert enc(type_id, torch.randn(3, 1), step_emb).shape == (3, D)
