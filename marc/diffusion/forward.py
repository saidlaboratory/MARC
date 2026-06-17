import torch

def corrupt(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    
    # Simple fix for the current setup:
    # Use torch.ones_like for t=0, else look up the value
    a_bar = torch.ones_like(t, dtype=torch.float32, device=x0.device)
    
    # Only index where t > 0
    mask = t > 0
    a_bar[mask] = alpha_bar_t.to(x0.device)[t[mask] - 1]
    
    # Reshape for broadcasting
    while len(a_bar.shape) < len(x0.shape):
        a_bar = a_bar.unsqueeze(-1)
        
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps