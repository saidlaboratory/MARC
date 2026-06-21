# marc/diffusion/solve.py
import torch
from .sample import ddim_step
from .guidance import guided_score

def solve(G, model, cas_engine, checker, steps, n_samples, guidance_scale):
    """
    G: The constraint graph definition
    model: The denoiser model (stub for now until Quang finishes training)
    """
    # Initialize n parallel noisy graphs: x_T ~ N(0, I)
    x_t = torch.randn((n_samples, G.num_variables)) 
    
    # Define DDIM step intervals (e.g., jumping from 1000 down to 0 in `steps` jumps)
    time_steps = torch.linspace(1000, 0, steps + 1, dtype=torch.long)
    
    x_best = None
    best_residual = float('inf')

    for i in range(steps):
        t = time_steps[i]
        t_prev = time_steps[i + 1]
        
        # 1. Unconditional noise prediction from the model
        unconditional_eps = model(G, x_t, t)
        
        # 2. Apply CAS Guidance
        alpha_t = model.alphas_cumprod[t]
        guided_eps = guided_score(x_t, unconditional_eps, alpha_t, cas_engine, guidance_scale)
        
        # 3. Take DDIM Step
        x_t = ddim_step(x_t, t, t_prev, guided_eps, model.alphas_cumprod)
        
        # 4. Early Stop & Checker Eval (Check every few steps to save compute)
        if i % 5 == 0 or i == steps - 1:
            for n in range(n_samples):
                sample_state = x_t[n]
                # If the exact checker accepts it, halt entirely.
                if checker.accepts(sample_state):
                    return sample_state
                
                # Best-of-N tracking
                residual = cas_engine.energy(sample_state)
                if residual < best_residual:
                    best_residual = residual
                    x_best = sample_state

    # Return the state with the lowest CAS residual if no exact match is found
    return x_best