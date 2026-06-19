import torch


def apply_guidance(
    score: torch.Tensor,
    x_vals: list,
    cas_engine,
    lambda_t: float = 1.0,
) -> torch.Tensor:
    """Subtract lambda_t * energy_grad from the score.

    The guided score is: s_guided = s_theta - lambda_t * grad_E(x)

    Args:
        score: [n, d] score estimate from denoiser
        x_vals: python list of current variable values (length n)
        cas_engine: CASEngine instance with .energy_grad(x_vals) -> list[float]
        lambda_t: guidance weight

    Returns:
        guided score tensor with same shape as score
    """
    grad = cas_engine.energy_grad(x_vals)
    grad_tensor = torch.tensor(grad, dtype=score.dtype, device=score.device).view_as(score)
    return score - lambda_t * grad_tensor
