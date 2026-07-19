import torch

from .sample import ddim_step
from .guidance import apply_guidance
from .schedule import cosine_beta_schedule
from marc.graph.pyg import build_heterodata


def solve(
    graph,
    denoiser,
    cas_engine,
    steps: int = 40,
    N: int = 8,
    guidance_weight: float = 1.0,
    eta: float = 0.0,
    T: int = 1000,
) -> torch.Tensor:
    """Best-of-N DDIM inference with CAS guidance.

    Args:
        graph: FactorGraph
        denoiser: callable (HeteroData, t_tensor) -> eps_hat [n, 1]
        cas_engine: CASEngine instance
        steps: number of DDIM denoising steps
        N: number of independent rollouts (best-of-N)
        guidance_weight: lambda for CAS energy gradient guidance
        eta: DDIM stochasticity (0=deterministic)
        T: training timesteps used to build the schedule

    Returns:
        best x found across all rollouts, shape [n_vars, 1]
    """
    _, alpha_bar = cosine_beta_schedule(T)
    n_vars = len(graph.variables)

    timesteps = [int(t) for t in torch.linspace(T - 1, 0, steps, dtype=torch.long).tolist()]

    best_x = None
    best_energy = float("inf")

    # Build static graph structure once; only variable features change each step
    base_data = build_heterodata(graph)

    for _ in range(N):
        x = torch.randn(n_vars, 1)

        for t in timesteps:
            base_data["variable"].x = x
            t_tensor = torch.tensor([t], dtype=torch.long)
            eps_hat = denoiser(base_data, t_tensor)

            abar_t = alpha_bar[t]
            x_vals = x.reshape(-1).tolist()
            grad = torch.tensor(
                cas_engine.energy_grad(x_vals), dtype=eps_hat.dtype, device=eps_hat.device
            ).view_as(eps_hat)
            # Clip the guidance gradient: the CAS energy gradient grows without bound
            # as x drifts, which otherwise feeds back into an exploding trajectory.
            gnorm = grad.norm()
            max_gnorm = 10.0
            if gnorm > max_gnorm:
                grad = grad * (max_gnorm / (gnorm + 1e-8))
            guided_eps = eps_hat + guidance_weight * grad * (1.0 - abar_t).sqrt()

            x = ddim_step(x, guided_eps, t, alpha_bar, eta=eta)
            # Keep the state in a sane range so a single bad step can't diverge.
            x = x.clamp(-100.0, 100.0)

            x_vals = x.reshape(-1).tolist()
            if cas_engine.accepts(x_vals):
                return x

        energy = cas_engine.energy(x.reshape(-1).tolist())
        if energy < best_energy:
            best_energy = energy
            best_x = x.clone()

    return best_x
