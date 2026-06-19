import torch


def ddim_step(
    x_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: int,
    alpha_bar: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM reverse step: compute x_{t-1} from x_t and predicted noise eps_hat.

    Args:
        x_t: [n, d] current noisy values
        eps_hat: [n, d] predicted noise from denoiser
        t: current timestep (0-indexed; t=T-1 is most noisy, t=0 is cleanest)
        alpha_bar: [T] cumulative product schedule
        eta: stochasticity (0=deterministic DDIM, 1=DDPM)

    Returns:
        x_{t-1}: [n, d] less-noisy values
    """
    abar_t = alpha_bar[t]
    x0_pred = (x_t - (1.0 - abar_t).sqrt() * eps_hat) / abar_t.sqrt()

    if t == 0:
        return x0_pred

    abar_prev = alpha_bar[t - 1]
    sigma_t = eta * ((1.0 - abar_prev) / (1.0 - abar_t)).sqrt() * (1.0 - abar_t / abar_prev).sqrt()
    direction = (1.0 - abar_prev - sigma_t ** 2).clamp(min=0.0).sqrt() * eps_hat
    noise = sigma_t * torch.randn_like(x_t) if eta > 0.0 else 0.0
    return abar_prev.sqrt() * x0_pred + direction + noise
