# marc/diffusion/guidance.py
import torch
import numpy as np

def guided_score(x_t, unconditional_eps, alpha_t, cas_engine, guidance_scale):
    """
    Modifies the noise prediction using the exact mathematical gradient.
    """
    # Get the gradient of the constraints' residuals from the CAS engine
    # cas_engine.energy_grad expects unpacked per-variable values.
    # We detach, to CPU, convert to numpy, and transpose to [n_vars, batch_size]
    x_vals_np = x_t.detach().cpu().numpy().T
    
    energy_gradient_list = cas_engine.energy_grad(x_vals_np)
    
    # Convert list of arrays back to a tensor of shape [batch_size, n_vars]
    energy_gradient = torch.tensor(np.array(energy_gradient_list), dtype=torch.float32, device=x_t.device).T
    
    # Standard diffusion formulation: subtract gradient of energy (gradient descent)
    # Since x_{t-1} = x_t - noise, adding to the noise subtracts from x.
    sigma_t = torch.sqrt(1 - alpha_t)
    guided_eps = unconditional_eps + (guidance_scale * sigma_t * energy_gradient)
    
    return guided_eps