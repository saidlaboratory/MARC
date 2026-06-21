# marc/diffusion/guidance.py
import torch

def guided_score(x_t, unconditional_eps, alpha_t, cas_engine, guidance_scale):
    """
    Modifies the noise prediction using the exact mathematical gradient.
    """
    # Get the gradient of the constraints' residuals from the CAS engine
    # cas_engine.energy_grad returns dE/dx where E is the constraint violation
    energy_gradient = cas_engine.energy_grad(x_t)
    
    # Standard diffusion formulation: subtract gradient of energy (gradient descent)
    # Since x_{t-1} = x_t - noise, adding to the noise subtracts from x.
    sigma_t = torch.sqrt(1 - alpha_t)
    guided_eps = unconditional_eps + (guidance_scale * sigma_t * energy_gradient)
    
    return guided_eps