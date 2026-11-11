from __future__ import annotations

import pytest
import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.transition_token_semantics import MatchedDeltaTokenEncoderProbe
from openpi.models_pytorch.transition_token_semantics import MatchedFutureDecoderProbe
from openpi.models_pytorch.transition_token_semantics import build_swap_indices
from openpi.models_pytorch.transition_token_semantics import delta_token_loss_per_sample
from openpi.models_pytorch.transition_token_semantics import masked_geometry_loss_per_sample


def _config(*, zero_init_decoder: bool = False) -> DeltaTokenizerConfig:
    return DeltaTokenizerConfig(
        teacher_dim=8,
        hidden_dim=16,
        delta_dim=4,
        num_delta_tokens=2,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        max_tokens=6,
        max_horizon=8,
        relation_max_tokens=6,
        zero_init_decoder=zero_init_decoder,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    current = torch.randn(3, 5, 8)
    future = torch.randn(3, 5, 8)
    delta = torch.randn(3, 2, 4)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    horizon = torch.tensor([2, 2, 2], dtype=torch.long)
    return current, future, delta, mask, horizon


@pytest.mark.parametrize("mode", ["current_only", "transition_only", "current_transition"])
def test_matched_future_decoder_probe_shapes(mode: str) -> None:
    current, _future, delta, mask, horizon = _inputs()
    probe = MatchedFutureDecoderProbe(_config(), mode)

    output = probe(current, delta, token_mask=mask, horizon_ids=horizon)

    assert output.shape == current.shape
    assert output.dtype == torch.float32


@pytest.mark.parametrize("mode", ["current_only", "future_only", "current_future"])
def test_matched_delta_token_probe_shapes(mode: str) -> None:
    current, future, _delta, mask, horizon = _inputs()
    probe = MatchedDeltaTokenEncoderProbe(_config(), mode)

    output = probe(current, future, token_mask=mask, horizon_ids=horizon)

    assert output.shape == (3, 2, 4)


def test_probe_bad_shapes_are_rejected() -> None:
    current, future, delta, mask, horizon = _inputs()
    decoder_probe = MatchedFutureDecoderProbe(_config(), "current_transition")
    token_probe = MatchedDeltaTokenEncoderProbe(_config(), "current_future")

    with pytest.raises(ValueError, match="delta_tokens must have shape"):
        decoder_probe(current, delta[:, :1], token_mask=mask, horizon_ids=horizon)
    with pytest.raises(ValueError, match="identical shape"):
        token_probe(current, future[:, :-1], token_mask=mask, horizon_ids=horizon)


def test_zero_initialized_decoder_probe_is_identity() -> None:
    current, _future, delta, mask, horizon = _inputs()
    probe = MatchedFutureDecoderProbe(_config(zero_init_decoder=True), "current_transition")

    output = probe(current, delta, token_mask=mask, horizon_ids=horizon)

    assert torch.equal(output, current.float())


def test_probe_gradient_flow_respects_selected_inputs() -> None:
    current, future, delta, mask, horizon = _inputs()
    current = current.requires_grad_()
    future = future.requires_grad_()
    delta = delta.requires_grad_()
    decoder_probe = MatchedFutureDecoderProbe(_config(), "transition_only")
    token_probe = MatchedDeltaTokenEncoderProbe(_config(), "future_only")

    decoder_output = decoder_probe(current, delta, token_mask=mask, horizon_ids=horizon)
    token_output = token_probe(current, future, token_mask=mask, horizon_ids=horizon)
    loss = decoder_output.square().mean() + token_output.square().mean()
    loss.backward()

    assert current.grad is None
    assert future.grad is not None
    assert future.grad.abs().sum() > 0
    assert delta.grad is not None
    assert delta.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in decoder_probe.parameters()
    )
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in token_probe.parameters())


def test_probe_determinism_with_fixed_initialization() -> None:
    current, future, delta, mask, horizon = _inputs()
    torch.manual_seed(19)
    decoder_a = MatchedFutureDecoderProbe(_config(), "current_transition").eval()
    torch.manual_seed(19)
    decoder_b = MatchedFutureDecoderProbe(_config(), "current_transition").eval()
    torch.manual_seed(23)
    encoder_a = MatchedDeltaTokenEncoderProbe(_config(), "current_future").eval()
    torch.manual_seed(23)
    encoder_b = MatchedDeltaTokenEncoderProbe(_config(), "current_future").eval()

    with torch.no_grad():
        decoder_output_a = decoder_a(current, delta, token_mask=mask, horizon_ids=horizon)
        decoder_output_b = decoder_b(current, delta, token_mask=mask, horizon_ids=horizon)
        encoder_output_a = encoder_a(current, future, token_mask=mask, horizon_ids=horizon)
        encoder_output_b = encoder_b(current, future, token_mask=mask, horizon_ids=horizon)

    assert torch.equal(decoder_output_a, decoder_output_b)
    assert torch.equal(encoder_output_a, encoder_output_b)


def test_losses_and_swap_indices() -> None:
    current, future, delta, mask, _horizon = _inputs()
    geometry = masked_geometry_loss_per_sample(current, future, mask)
    token = delta_token_loss_per_sample(delta, torch.zeros_like(delta))
    task_ids = torch.tensor([0, 0, 1, 1, 1, 2])
    episode_ids = torch.tensor([10, 11, 20, 20, 21, 30])

    same_task, same_task_valid = build_swap_indices(
        task_ids,
        secondary_ids=episode_ids,
        relation="same_primary_different_secondary",
        seed=3,
    )
    cross_task, cross_task_valid = build_swap_indices(task_ids, relation="different_primary", seed=3)
    same_episode, same_episode_valid = build_swap_indices(episode_ids, relation="same_primary", seed=3)

    assert geometry.shape == (3,)
    assert torch.isfinite(geometry).all()
    assert token.shape == (3,)
    assert torch.isfinite(token).all()
    assert same_task_valid.tolist() == [True, True, True, True, True, False]
    assert torch.equal(task_ids[same_task[same_task_valid]], task_ids[same_task_valid])
    assert torch.all(episode_ids[same_task[same_task_valid]] != episode_ids[same_task_valid])
    assert cross_task_valid.all()
    assert torch.all(task_ids[cross_task] != task_ids)
    assert same_episode_valid.tolist() == [False, False, True, True, False, False]
    assert torch.equal(episode_ids[same_episode[same_episode_valid]], episode_ids[same_episode_valid])


def test_matched_probes_tiny_overfit() -> None:
    torch.manual_seed(29)
    config = _config()
    current = torch.randn(4, 5, 8)
    future = current + 0.15 * torch.tanh(current)
    delta = torch.randn(4, 2, 4)
    target_delta = torch.stack(
        [future.mean(dim=1)[:, :4], future.square().mean(dim=1)[:, :4]],
        dim=1,
    )
    mask = torch.ones(4, 5, dtype=torch.bool)
    horizon = torch.full((4,), 2, dtype=torch.long)
    decoder_probe = MatchedFutureDecoderProbe(config, "current_transition")
    token_probe = MatchedDeltaTokenEncoderProbe(config, "current_future")
    optimizer = torch.optim.AdamW(
        [*decoder_probe.parameters(), *token_probe.parameters()],
        lr=1e-2,
        weight_decay=0.0,
    )
    losses = []

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        pred_future = decoder_probe(current, delta, token_mask=mask, horizon_ids=horizon)
        pred_delta = token_probe(current, future, token_mask=mask, horizon_ids=horizon)
        loss = masked_geometry_loss_per_sample(pred_future, future, mask).mean()
        loss = loss + delta_token_loss_per_sample(pred_delta, target_delta).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.35
