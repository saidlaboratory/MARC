import torch

def inject_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)