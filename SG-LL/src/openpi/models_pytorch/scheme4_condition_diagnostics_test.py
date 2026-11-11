import pytest
import torch

from openpi.models_pytorch.scheme4_condition_diagnostics import delta_token_distance
from openpi.models_pytorch.scheme4_condition_diagnostics import margin_negative_condition_loss
from openpi.models_pytorch.scheme4_condition_diagnostics import roll_batch_condition


def test_roll_batch_condition_shapes_and_validity() -> None:
    value = torch.arange(12).view(3, 4)
    rolled, valid = roll_batch_condition(value)

    assert valid.item() is True
    assert rolled.shape == value.shape
    assert torch.equal(rolled[0], value[-1])
    assert torch.equal(rolled[1], value[0])


def test_roll_batch_condition_singleton_is_invalid_identity_copy() -> None:
    value = torch.randn(1, 2, 3)
    rolled, valid = roll_batch_condition(value)

    assert valid.item() is False
    assert rolled.shape == value.shape
    assert torch.allclose(rolled, value)
    assert rolled.data_ptr() != value.data_ptr()


def test_delta_token_distance_shape_check_and_gradient() -> None:
    pred = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4)

    loss = delta_token_distance(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="identical shape"):
        delta_token_distance(pred, target[:, :-1])


def test_margin_negative_condition_loss() -> None:
    normal = torch.tensor(0.2, requires_grad=True)
    negative = torch.tensor(0.1, requires_grad=True)
    valid = torch.tensor(True)

    loss = margin_negative_condition_loss(normal, negative, margin=0.05, valid=valid)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.tensor(0.15))
    assert normal.grad is not None and normal.grad.item() > 0
    assert negative.grad is not None and negative.grad.item() < 0


def test_margin_negative_condition_loss_invalid_batch_zeroes() -> None:
    normal = torch.tensor(0.2, requires_grad=True)
    negative = torch.tensor(0.1, requires_grad=True)
    valid = torch.tensor(False)

    loss = margin_negative_condition_loss(normal, negative, margin=0.05, valid=valid)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.tensor(0.0))
    assert normal.grad is not None and normal.grad.item() == 0.0
    assert negative.grad is not None and negative.grad.item() == 0.0
