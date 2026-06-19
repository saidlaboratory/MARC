import torch
import pytest
from marc.model.structure_head import StructureHead


def test_structure_head_output_shapes():
    D = 64
    num_types = 5  # ABSENT + 4 real types
    n_slots = 10   # padded to max slots

    head = StructureHead(D=D, num_slot_types=num_types)
    h_v = torch.randn(n_slots, D)
    eps_hat, slot_logits = head(h_v)

    assert eps_hat.shape == (n_slots, 1), f"Expected ({n_slots}, 1), got {eps_hat.shape}"
    assert slot_logits.shape == (n_slots, num_types), \
        f"Expected ({n_slots}, {num_types}), got {slot_logits.shape}"


def test_structure_head_absent_type():
    head = StructureHead(D=32, num_slot_types=4)
    assert head.ABSENT_TYPE == 0


def test_structure_loss_shape():
    head = StructureHead(D=32, num_slot_types=4)
    h_v = torch.randn(6, 32)
    _, slot_logits = head(h_v)
    target = torch.randint(0, 4, (6,))
    loss = head.structure_loss(slot_logits, target)
    assert loss.shape == (), "Structure loss should be scalar"
    assert loss.item() >= 0.0


def test_value_loss_masks_absent():
    head = StructureHead(D=32, num_slot_types=4)
    n_slots = 5
    eps_hat = torch.ones(n_slots, 1)
    eps_true = torch.zeros(n_slots, 1)
    # Mark slots 3,4 as ABSENT (type=0)
    slot_types = torch.tensor([1, 2, 3, 0, 0])
    loss_with_absent = head.value_loss(eps_hat, eps_true, slot_types)

    # Loss should only be over non-absent slots (0,1,2)
    # If all 5 contributed: loss = 1.0; if only 3: also 1.0 (MSE per element)
    # The key test: ABSENT slots produce 0 * (1-0)^2 = 0 contribution
    assert loss_with_absent.item() >= 0.0

    # All absent: loss should be 0
    all_absent = torch.zeros(n_slots, dtype=torch.long)
    loss_all_absent = head.value_loss(eps_hat, eps_true, all_absent)
    assert loss_all_absent.item() == pytest.approx(0.0)


def test_gradient_flows_through_head():
    head = StructureHead(D=32, num_slot_types=4)
    h_v = torch.randn(5, 32, requires_grad=True)
    eps_hat, slot_logits = head(h_v)
    (eps_hat.sum() + slot_logits.sum()).backward()
    assert h_v.grad is not None
    assert not torch.isnan(h_v.grad).any()


def test_structure_head_batch_dim_preserved():
    head = StructureHead(D=16, num_slot_types=3)
    for n in [1, 5, 10, 20]:
        h_v = torch.randn(n, 16)
        eps_hat, logits = head(h_v)
        assert eps_hat.shape[0] == n
        assert logits.shape[0] == n


def test_structure_head_different_configs():
    for D, num_types in [(32, 3), (128, 8), (256, 16)]:
        head = StructureHead(D=D, num_slot_types=num_types)
        h_v = torch.randn(4, D)
        eps_hat, logits = head(h_v)
        assert eps_hat.shape == (4, 1)
        assert logits.shape == (4, num_types)
