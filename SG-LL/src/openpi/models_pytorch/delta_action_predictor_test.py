import dataclasses
import io

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictor
from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictorConfig
from openpi.models_pytorch.delta_action_predictor import delta_action_predictor_loss


def _small_config(*, zero_init_output: bool = False) -> DeltaActionPredictorConfig:
    return DeltaActionPredictorConfig(
        teacher_dim=12,
        delta_dim=6,
        num_delta_tokens=3,
        action_horizon=4,
        action_dim=5,
        state_dim=7,
        hidden_dim=24,
        num_layers=2,
        dropout=0.0,
        max_horizon=8,
        zero_init_output=zero_init_output,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    config = _small_config()
    batch_size = 2
    num_tokens = 5
    current = torch.randn(batch_size, num_tokens, config.teacher_dim)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
    state = torch.randn(batch_size, config.state_dim)
    token_mask = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    horizon_ids = torch.tensor([1, 3], dtype=torch.long)
    target_delta = torch.randn(batch_size, config.num_delta_tokens, config.delta_dim)
    return current, actions, state, token_mask, horizon_ids, target_delta


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_delta_action_predictor_shapes() -> None:
    predictor = DeltaActionPredictor(_small_config())
    current, actions, state, token_mask, horizon_ids, _target_delta = _inputs()

    pred_delta = predictor(current, actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    assert pred_delta.shape == (2, 3, 6)
    assert pred_delta.dtype == torch.float32


def test_delta_action_predictor_rejects_bad_shapes() -> None:
    predictor = DeltaActionPredictor(_small_config())
    current, actions, state, token_mask, horizon_ids, target_delta = _inputs()

    with pytest.raises(ValueError, match="current_tokens"):
        predictor(current[..., :-1], actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    with pytest.raises(ValueError, match="actions"):
        predictor(current, actions[:, :-1], state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    with pytest.raises(ValueError, match="state"):
        predictor(current, actions, state=state[:, :-1], token_mask=token_mask, horizon_ids=horizon_ids)

    with pytest.raises(ValueError, match="token_mask"):
        predictor(current, actions, state=state, token_mask=token_mask[:, :-1], horizon_ids=horizon_ids)

    with pytest.raises(ValueError, match="horizon_ids"):
        predictor(current, actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids[:, None])

    with pytest.raises(ValueError, match="target_delta_tokens"):
        delta_action_predictor_loss(
            predictor,
            current,
            actions,
            target_delta[:, :-1],
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )


def test_delta_action_predictor_gradient_flow() -> None:
    predictor = DeltaActionPredictor(_small_config())
    current, actions, state, token_mask, horizon_ids, target_delta = _inputs()
    current = current.requires_grad_()

    loss_output = delta_action_predictor_loss(
        predictor,
        current,
        actions,
        target_delta.detach(),
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )
    loss_output.loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert _has_nonzero_grad(predictor)


def test_delta_action_predictor_action_sensitivity() -> None:
    predictor = DeltaActionPredictor(_small_config()).eval()
    current, actions, state, token_mask, horizon_ids, _target_delta = _inputs()
    changed_actions = actions.clone()
    changed_actions[0, 0, 0] = changed_actions[0, 0, 0] + 10.0

    with torch.no_grad():
        pred_a = predictor(current, actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)
        pred_b = predictor(current, changed_actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    assert not torch.allclose(pred_a, pred_b)


def test_delta_action_predictor_checkpoint_roundtrip() -> None:
    torch.manual_seed(1)
    config = _small_config()
    predictor = DeltaActionPredictor(config).eval()
    current, actions, state, token_mask, horizon_ids, _target_delta = _inputs()
    with torch.no_grad():
        pred_before = predictor(current, actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    buffer = io.BytesIO()
    torch.save({"config": predictor.config_dict(), "state_dict": predictor.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)

    restored = DeltaActionPredictor(DeltaActionPredictorConfig(**payload["config"])).eval()
    restored.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        pred_after = restored(current, actions, state=state, token_mask=token_mask, horizon_ids=horizon_ids)

    assert torch.allclose(pred_before, pred_after)


def test_delta_action_predictor_tiny_overfit_smoke() -> None:
    torch.manual_seed(2)
    config = dataclasses.replace(_small_config(), hidden_dim=32, num_layers=3)
    predictor = DeltaActionPredictor(config)
    batch_size = 16
    num_tokens = 5
    current = torch.randn(batch_size, num_tokens, config.teacher_dim)
    actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
    state = torch.randn(batch_size, config.state_dim)
    horizon_ids = torch.arange(batch_size, dtype=torch.long) % config.max_horizon

    condition = torch.cat([current.mean(dim=1), actions.reshape(batch_size, -1), state], dim=-1)
    target_proj = torch.randn(condition.shape[-1], config.num_delta_tokens * config.delta_dim)
    target_delta = torch.tanh(condition @ target_proj * 0.1).reshape(
        batch_size, config.num_delta_tokens, config.delta_dim
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss_output = delta_action_predictor_loss(
            predictor,
            current,
            actions,
            target_delta,
            state=state,
            horizon_ids=horizon_ids,
            cosine_weight=0.0,
        )
        loss_output.loss.backward()
        optimizer.step()
        losses.append(float(loss_output.loss.detach()))

    with torch.no_grad():
        shuffled = actions.flip(0)
        normal = predictor(current, actions, state=state, horizon_ids=horizon_ids)
        shuffled_pred = predictor(current, shuffled, state=state, horizon_ids=horizon_ids)
        normal_loss = F.mse_loss(normal, target_delta)
        shuffled_loss = F.mse_loss(shuffled_pred, target_delta)

    assert losses[-1] < losses[0] * 0.5
    assert shuffled_loss > normal_loss
