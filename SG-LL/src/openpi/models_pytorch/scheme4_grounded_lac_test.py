import pytest
import torch

from openpi.models_pytorch.scheme4_grounded_lac_pytorch import Scheme4GroundedDecoderActionBridge


def test_grounded_decoder_action_bridge_shape_and_zero_init() -> None:
    bridge = Scheme4GroundedDecoderActionBridge(action_intent_dim=6, action_horizon=4, action_dim=3)
    action_intent = torch.randn(2, 6)

    out = bridge(action_intent)

    assert out.shape == (2, 4, 3)
    assert torch.allclose(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


def test_grounded_decoder_action_bridge_rejects_bad_shape() -> None:
    bridge = Scheme4GroundedDecoderActionBridge(action_intent_dim=6, action_horizon=4, action_dim=3)

    with pytest.raises(ValueError, match="action_intent"):
        bridge(torch.randn(2, 5))

    with pytest.raises(ValueError, match="action_intent"):
        bridge(torch.randn(2, 1, 6))


def test_grounded_decoder_action_bridge_grad_flow_after_open_init() -> None:
    bridge = Scheme4GroundedDecoderActionBridge(action_intent_dim=6, action_horizon=4, action_dim=3)
    bridge.proj[-1].weight.data.normal_(mean=0.0, std=0.01)
    action_intent = torch.randn(2, 6, requires_grad=True)
    target = torch.randn(2, 4, 3)

    loss = (bridge(action_intent) - target).square().mean()
    loss.backward()

    assert action_intent.grad is not None
    assert action_intent.grad.abs().sum() > 0
    assert bridge.proj[-1].weight.grad is not None
    assert bridge.proj[-1].weight.grad.abs().sum() > 0
