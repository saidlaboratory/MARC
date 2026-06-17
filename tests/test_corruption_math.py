import torch
from marc.diffusion.schedule import cosine_beta_schedule
from marc.diffusion.forward import corrupt

def test_corruption_math():
    T = 1000
    batch_size = 10
    _, alpha_bar_t = cosine_beta_schedule(T)
    
    x0 = torch.randn((batch_size, 1))
    eps = torch.randn_like(x0)
    t = torch.tensor([500] * batch_size)
    
    x_t = corrupt(x0, t, eps, alpha_bar_t)
    
    assert x_t.shape == x0.shape
    assert not torch.allclose(x0, x_t)

    t_zero = torch.tensor([0] * batch_size)
    x_t_zero = corrupt(x0, t_zero, eps, alpha_bar_t)
    # At t=0, alpha_bar is 1.0, so x_t should be x0
    assert torch.allclose(x0, x_t_zero, atol=1e-5)