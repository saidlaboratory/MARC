# pyrefly: ignore [missing-import]
import torch
import math
from typing import Tuple

def cosine_beta_schedule(timesteps: int = 1000, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alpha_bar_t = alphas_cumprod[1:]
    
    return betas, alpha_bar_t