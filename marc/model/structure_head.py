import torch
import torch.nn as nn


class StructureHead(nn.Module):
    """Categorical output head for structure diffusion (D3PM over slot types).

    Extends the value-diffusion output by adding a categorical prediction
    for slot types (variable/factor node types including ABSENT).

    Design (P3 preliminary):
    - Takes variable embeddings h_v: [n_slots, D] from upstream GNN
    - Outputs both continuous eps_hat [n_slots, 1] AND slot type logits [n_slots, num_types]
    - ABSENT slots: mask their contribution to the value-diffusion loss during training

    For structure denoising per D3PM:
    - Forward: categorical noise corrupts slot types toward ABSENT
    - Reverse: this head predicts p(c_0 | c_t) per slot
    """

    ABSENT_TYPE = 0  # Type index 0 = ABSENT (slot doesn't exist in this problem)

    def __init__(self, D: int, num_slot_types: int, num_var_types: int = 4):
        """
        Args:
            D: embedding dimension (matches the upstream GNN's hidden dim)
            num_slot_types: number of categorical slot types (including ABSENT at index 0)
            num_var_types: number of variable type embeddings for input
        """
        super().__init__()
        self.D = D
        self.num_slot_types = num_slot_types

        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, 1),
        )

        self.structure_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, num_slot_types),
        )

    def forward(self, h_v: torch.Tensor) -> tuple:
        """
        Args:
            h_v: [n_slots, D] variable embeddings from upstream GNN

        Returns:
            (eps_hat, slot_logits):
              - eps_hat: [n_slots, 1] predicted continuous noise per slot
              - slot_logits: [n_slots, num_slot_types] unnormalized logits for slot types
        """
        eps_hat = self.value_head(h_v)
        slot_logits = self.structure_head(h_v)
        return eps_hat, slot_logits

    def structure_loss(
        self,
        slot_logits: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for slot type prediction (D3PM denoising).

        All slots (including ABSENT) contribute equally to the loss here.
        During training, the caller may mask out ABSENT slots in target_types.
        """
        return nn.functional.cross_entropy(slot_logits, target_types)

    def value_loss(
        self,
        eps_hat: torch.Tensor,
        eps_true: torch.Tensor,
        slot_types: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss for value diffusion, masked to non-ABSENT slots only."""
        mask = (slot_types != self.ABSENT_TYPE).float().unsqueeze(-1)
        return nn.functional.mse_loss(eps_hat * mask, eps_true * mask)
