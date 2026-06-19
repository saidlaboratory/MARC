import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from marc.diffusion.forward import corrupt


def train_step_A(
    denoiser: nn.Module,
    batch,
    alpha_bar: torch.Tensor,
    T: int,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """Single training step for Stage A denoising pretraining (DSM loss).

    Args:
        denoiser: model with signature (HeteroData, t: Tensor) -> eps_hat [n_vars, 1]
        batch: tuple of (batched_data, solutions) where solutions is a list of [n_i, 1] tensors
        alpha_bar: cosine schedule cumulative products [T]
        T: number of diffusion timesteps
        optimizer: PyTorch optimizer
        device: device string

    Returns:
        loss value (float)
    """
    denoiser.train()
    optimizer.zero_grad()

    data, solutions = batch
    data = data.to(device)
    alpha_bar = alpha_bar.to(device)

    total_loss = torch.tensor(0.0, device=device)
    n_problems = len(solutions)

    # Unpack batched graph into per-problem graphs so each denoiser call
    # sees exactly n_vars_i nodes matching its paired solution tensor.
    try:
        graphs = data.to_data_list()
    except AttributeError:
        # data is already a single HeteroData (batch_size=1 path)
        graphs = [data] * n_problems

    for graph, x0 in zip(graphs, solutions):
        x0 = x0.to(device)
        t = torch.randint(1, T + 1, (1,), device=device)
        eps = torch.randn_like(x0)
        x_t = corrupt(x0, t, eps, alpha_bar)
        eps_hat = denoiser(graph, t)
        loss = nn.functional.mse_loss(eps_hat, eps)
        total_loss = total_loss + loss

    avg_loss = total_loss / n_problems
    avg_loss.backward()
    optimizer.step()
    return avg_loss.item()


def train_stage_a(
    denoiser: nn.Module,
    dataloader: DataLoader,
    alpha_bar: torch.Tensor,
    T: int = 1000,
    epochs: int = 10,
    checkpoint_dir: str = "checkpoints/stage_a",
    lr: float = 1e-3,
    device: str = "cpu",
    use_iterative_loss: bool = False,
) -> nn.Module:
    """Full Stage A training loop with epoch checkpointing.

    Saves a checkpoint at the end of each epoch to checkpoint_dir/epoch_{i}.pt.
    If use_iterative_loss=True, also adds an energy-reduction term (requires
    cas_engine in denoiser).

    Returns:
        The trained denoiser.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    denoiser = denoiser.to(device)

    for epoch in range(epochs):
        epoch_losses = []
        for batch in dataloader:
            loss = train_step_A(denoiser, batch, alpha_bar, T, optimizer, device)
            epoch_losses.append(loss)

        avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg:.4f}")

        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": denoiser.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg,
            },
            ckpt_path,
        )

    return denoiser


def run_ablation(
    denoiser: nn.Module,
    problems,
    noise_on: bool = True,
    steps: int = 20,
    sigma: float = 0.1,
    device: str = "cpu",
) -> dict:
    """Entrapment ablation: run inference with/without noise injection.

    For each problem, run `steps` iterative refinement steps and track whether
    the run gets trapped at a local minimum.

    Args:
        denoiser: trained model
        problems: list of (FactorGraph, solution_dict) pairs
        noise_on: whether to inject Langevin noise during refinement
        steps: number of iterative refinement steps
        sigma: noise magnitude when noise_on=True
        device: device string

    Returns:
        dict with keys: "trapped_count", "total", "entrapment_rate"
    """
    from marc.diffusion.schedule import cosine_beta_schedule
    from marc.graph.pyg import build_heterodata

    denoiser.eval()
    _, alpha_bar = cosine_beta_schedule(1000)

    trapped = 0
    with torch.no_grad():
        for graph, solution in problems:
            data = build_heterodata(graph)
            n_vars = len(graph.variables)
            x = torch.randn(n_vars, 1, device=device)
            prev_x = x.clone()
            stalled = False

            for step in range(steps):
                t = torch.tensor([max(1, steps - step)], device=device)
                eps_hat = denoiser(data, t)
                x = x - 0.1 * eps_hat
                if noise_on:
                    x = x + sigma * torch.randn_like(x)
                if step > 5 and (x - prev_x).norm() < 1e-4:
                    stalled = True
                    break
                prev_x = x.clone()

            if stalled:
                trapped += 1

    return {
        "trapped_count": trapped,
        "total": len(problems),
        "entrapment_rate": trapped / len(problems) if problems else 0.0,
    }
