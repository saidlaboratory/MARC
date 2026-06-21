# marc/diffusion/sample.py
import torch

def ddim_step(x_t, t, t_prev, model_output, alphas_cumprod, eta=0.0):
    """
    x_t: Current graph variable states
    model_output: Epsilon predicted by the denoiser
    """
    alpha_t = alphas_cumprod[t]
    alpha_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
    
    # 1. Predict the denoised state (x_0)
    pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
    
    # 2. Determine direction pointing to x_t
    # (Setting eta=0.0 makes the sampling deterministic)
    dir_xt = torch.sqrt(1 - alpha_t_prev) * model_output
    
    # 3. Compute the previous noisy state
    x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
    return x_t_prev