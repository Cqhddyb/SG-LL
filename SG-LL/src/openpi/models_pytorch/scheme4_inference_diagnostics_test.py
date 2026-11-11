import pytest
import torch

from openpi.models_pytorch.scheme4_inference_diagnostics import STEP_METRICS
from openpi.models_pytorch.scheme4_inference_diagnostics import finalize_request_diagnostics
from openpi.models_pytorch.scheme4_inference_diagnostics import flow_step_diagnostics


def _step(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    hidden = torch.randn(batch_size, 4, 6)
    residual = torch.randn(batch_size, 4, 6) * 0.1
    base_velocity = torch.randn(batch_size, 4, 7)
    full_velocity = base_velocity + torch.randn(batch_size, 4, 7) * 0.05
    return flow_step_diagnostics(hidden, residual, full_velocity, base_velocity)


def test_flow_step_diagnostics_shapes_and_finite_values() -> None:
    metrics = _step()
    assert tuple(metrics) == STEP_METRICS
    assert all(value.shape == (2,) for value in metrics.values())
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert all(not value.requires_grad for value in metrics.values())


def test_flow_step_diagnostics_rejects_implicit_broadcasting() -> None:
    hidden = torch.randn(2, 4, 6)
    residual = torch.randn(1, 4, 6)
    velocity = torch.randn(2, 4, 7)
    with pytest.raises(ValueError, match="exactly match"):
        flow_step_diagnostics(hidden, residual, velocity, velocity)
    with pytest.raises(ValueError, match=r"share exact \[B,H\]"):
        flow_step_diagnostics(hidden, torch.zeros_like(hidden), velocity[:, :3], velocity[:, :3])


def test_finalize_request_diagnostics_contract_and_determinism() -> None:
    steps = [_step(), _step(), _step()]
    pred_delta = torch.ones(2, 3, 8)
    action_intent = torch.ones(2, 6)
    first = finalize_request_diagnostics(
        steps,
        pred_delta_tokens=pred_delta,
        action_intent=action_intent,
        fusion_scale=1.0,
    )
    second = finalize_request_diagnostics(
        steps,
        pred_delta_tokens=pred_delta,
        action_intent=action_intent,
        fusion_scale=1.0,
    )
    assert all(first[name].shape == (2, 3) for name in STEP_METRICS)
    assert first["pred_delta_token_l2_mean"].shape == (2,)
    assert first["action_intent_l2"].shape == (2,)
    assert torch.equal(first["fusion_scale"], torch.ones(2))
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_finalize_request_diagnostics_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        finalize_request_diagnostics(
            [_step()],
            pred_delta_tokens=torch.randn(2, 3, 8),
            action_intent=torch.randn(1, 6),
            fusion_scale=1.0,
        )
