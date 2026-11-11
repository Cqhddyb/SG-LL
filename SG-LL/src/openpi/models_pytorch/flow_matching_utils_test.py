import pytest
import torch

from openpi.models_pytorch.flow_matching_utils import estimate_action_endpoint


def test_estimate_action_endpoint_shape_and_formula() -> None:
    noisy_actions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    timestep = torch.tensor([0.25, 0.5])
    velocity = torch.ones_like(noisy_actions) * 2.0

    action_hat = estimate_action_endpoint(noisy_actions, timestep, velocity)

    expected = torch.tensor(
        [
            [[0.5, 1.5], [2.5, 3.5]],
            [[4.0, 5.0], [6.0, 7.0]],
        ]
    )
    assert action_hat.shape == noisy_actions.shape
    assert torch.allclose(action_hat, expected)


def test_estimate_action_endpoint_rejects_bad_shapes() -> None:
    noisy_actions = torch.randn(2, 4, 3)
    timestep = torch.rand(2)
    velocity = torch.randn(2, 4, 3)

    with pytest.raises(ValueError, match="noisy_actions"):
        estimate_action_endpoint(noisy_actions[:, 0], timestep, velocity)

    with pytest.raises(ValueError, match="velocity shape"):
        estimate_action_endpoint(noisy_actions, timestep, velocity[:, :-1])

    with pytest.raises(ValueError, match="timestep"):
        estimate_action_endpoint(noisy_actions, timestep[:, None], velocity)


def test_estimate_action_endpoint_detaches_by_default() -> None:
    noisy_actions = torch.randn(2, 4, 3, requires_grad=True)
    timestep = torch.rand(2, requires_grad=True)
    velocity = torch.randn(2, 4, 3, requires_grad=True)

    action_hat = estimate_action_endpoint(noisy_actions, timestep, velocity)

    assert not action_hat.requires_grad


def test_estimate_action_endpoint_can_preserve_gradient_flow() -> None:
    noisy_actions = torch.randn(2, 4, 3, requires_grad=True)
    timestep = torch.rand(2, requires_grad=True)
    velocity = torch.randn(2, 4, 3, requires_grad=True)

    action_hat = estimate_action_endpoint(noisy_actions, timestep, velocity, detach=False)
    loss = action_hat.square().mean()
    loss.backward()

    assert noisy_actions.grad is not None
    assert timestep.grad is not None
    assert velocity.grad is not None
    assert noisy_actions.grad.abs().sum() > 0
    assert timestep.grad.abs().sum() > 0
    assert velocity.grad.abs().sum() > 0


def test_estimate_action_endpoint_is_deterministic() -> None:
    torch.manual_seed(7)
    noisy_actions = torch.randn(3, 2, 5)
    timestep = torch.rand(3)
    velocity = torch.randn(3, 2, 5)

    first = estimate_action_endpoint(noisy_actions, timestep, velocity)
    second = estimate_action_endpoint(noisy_actions, timestep, velocity)

    assert torch.equal(first, second)
