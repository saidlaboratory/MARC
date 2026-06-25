import torch
import pytest
from marc.diffusion.sample import ddim_step
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.guidance import apply_guidance


def test_ddim_step_shape():
    _, alpha_bar = cosine_beta_schedule(100)
    x_t = torch.randn(2, 1)
    eps_hat = torch.randn(2, 1)
    x_prev = ddim_step(x_t, eps_hat, t=50, alpha_bar=alpha_bar, eta=0.0)
    assert x_prev.shape == x_t.shape


def test_ddim_step_final():
    # At t=0, should return x0_pred (not crash)
    _, alpha_bar = cosine_beta_schedule(100)
    x_t = torch.randn(3, 1)
    eps_hat = torch.zeros(3, 1)  # zero noise => x_prev ≈ x_t / sqrt(abar_0)
    x_prev = ddim_step(x_t, eps_hat, t=0, alpha_bar=alpha_bar, eta=0.0)
    assert x_prev.shape == (3, 1)
    assert not torch.isnan(x_prev).any()


def test_ddim_deterministic(seed=42):
    # eta=0 => same input => same output
    _, alpha_bar = cosine_beta_schedule(100)
    x_t = torch.randn(2, 1)
    eps_hat = torch.randn(2, 1)
    r1 = ddim_step(x_t, eps_hat, t=30, alpha_bar=alpha_bar, eta=0.0)
    r2 = ddim_step(x_t, eps_hat, t=30, alpha_bar=alpha_bar, eta=0.0)
    assert torch.allclose(r1, r2)


def test_guidance_reduces_energy():
    # With a simple mock CAS: energy = 0.5*(x[0]^2 + x[1]^2), grad = x
    class MockCAS:
        def energy_grad(self, x_vals):
            return x_vals  # grad of 0.5*sum(x^2) is x

    cas = MockCAS()
    score = torch.zeros(2, 1)
    x_vals = [2.0, 3.0]
    guided = apply_guidance(score, x_vals, cas, lambda_t=1.0)
    # Should subtract [2.0, 3.0] from zeros
    assert guided[0].item() == pytest.approx(-2.0, abs=1e-5)
    assert guided[1].item() == pytest.approx(-3.0, abs=1e-5)
