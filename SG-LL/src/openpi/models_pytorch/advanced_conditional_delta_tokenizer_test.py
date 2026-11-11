import torch

from openpi.models_pytorch.advanced_conditional_delta_tokenizer import AdvancedConditionalDeltaTokenizer
from openpi.models_pytorch.advanced_conditional_delta_tokenizer import AdvancedConditionalDeltaTokenizerConfig
from openpi.models_pytorch.advanced_conditional_delta_tokenizer import advanced_conditional_delta_tokenizer_loss


def _config(*, zero_init_decoder: bool = True, condition_dropout_prob: float = 0.0):
    return AdvancedConditionalDeltaTokenizerConfig(
        teacher_dim=12,
        language_dim=7,
        action_horizon=4,
        action_dim=3,
        state_dim=5,
        hidden_dim=16,
        delta_dim=6,
        num_delta_tokens=2,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        condition_dropout_prob=condition_dropout_prob,
        max_tokens=8,
        max_language_tokens=6,
        max_horizon=16,
        relation_max_tokens=4,
        zero_init_decoder=zero_init_decoder,
    )


def _inputs():
    torch.manual_seed(0)
    current = torch.randn(2, 5, 12)
    future = current + 0.1 * torch.randn(2, 5, 12)
    language_tokens = torch.randn(2, 4, 7)
    language_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 5)
    token_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    horizon_ids = torch.tensor([3, 4], dtype=torch.long)
    return current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids


def test_advanced_conditional_delta_tokenizer_shapes() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config())
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()

    output = tokenizer(
        current,
        future,
        language_token_features=language_tokens,
        language_token_mask=language_mask,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )

    assert output.delta_tokens.shape == (2, 2, 6)
    assert output.future_recon.shape == current.shape


def test_advanced_conditional_delta_tokenizer_rejects_bad_shapes() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config())
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()

    try:
        tokenizer(
            current,
            future,
            language_token_features=language_tokens[:, :, :-1],
            language_token_mask=language_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "language_token_features" in str(exc)
    else:
        raise AssertionError("bad language token shape was accepted")

    try:
        tokenizer(
            current,
            future,
            language_token_features=language_tokens,
            language_token_mask=language_mask[:, :-1],
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "language_token_mask" in str(exc)
    else:
        raise AssertionError("bad language mask shape was accepted")

    try:
        tokenizer(
            current,
            future,
            language_token_features=language_tokens,
            language_token_mask=language_mask,
            actions=actions[:, :, :-1],
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "actions" in str(exc)
    else:
        raise AssertionError("bad actions shape was accepted")


def test_advanced_conditional_zero_delta_identity_with_zero_init_decoder() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config(zero_init_decoder=True))
    current, _future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()
    zero_delta = torch.zeros(2, 2, 6)

    recon = tokenizer.decode(
        current,
        zero_delta,
        language_token_features=language_tokens,
        language_token_mask=language_mask,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )

    assert torch.allclose(recon, current.float(), atol=0.0, rtol=0.0)


def test_advanced_conditional_delta_tokenizer_gradients_reach_conditions() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config())
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()
    language_tokens.requires_grad_(True)
    actions.requires_grad_(True)
    state.requires_grad_(True)

    output = tokenizer(
        current,
        future,
        language_token_features=language_tokens,
        language_token_mask=language_mask,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )
    loss = output.delta_tokens.square().mean() + output.future_recon.square().mean()
    loss.backward()

    assert language_tokens.grad is not None and language_tokens.grad.abs().sum() > 0
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0


def test_advanced_conditional_delta_tokenizer_deterministic_eval() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config(condition_dropout_prob=0.5)).eval()
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()

    with torch.no_grad():
        output_a = tokenizer(
            current,
            future,
            language_token_features=language_tokens,
            language_token_mask=language_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
        output_b = tokenizer(
            current,
            future,
            language_token_features=language_tokens,
            language_token_mask=language_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )

    assert torch.allclose(output_a.delta_tokens, output_b.delta_tokens)
    assert torch.allclose(output_a.future_recon, output_b.future_recon)


def test_advanced_conditional_delta_tokenizer_negative_condition_loss() -> None:
    tokenizer = AdvancedConditionalDeltaTokenizer(_config(zero_init_decoder=False))
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()

    loss_output = advanced_conditional_delta_tokenizer_loss(
        tokenizer,
        current,
        future,
        language_token_features=language_tokens,
        language_token_mask=language_mask,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        relation_weight=0.0,
        negative_condition_weight=0.5,
        negative_condition_margin=0.05,
    )

    assert loss_output.loss.ndim == 0
    assert "advanced_conditional_negative_condition_loss" in loss_output.metrics
    assert "advanced_conditional_neg_language_rec_loss" in loss_output.metrics
    assert "advanced_conditional_neg_action_rec_loss" in loss_output.metrics


def test_advanced_conditional_delta_tokenizer_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    tokenizer = AdvancedConditionalDeltaTokenizer(_config(zero_init_decoder=False))
    current, future, language_tokens, language_mask, actions, state, token_mask, horizon_ids = _inputs()
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss_output = advanced_conditional_delta_tokenizer_loss(
            tokenizer,
            current,
            future,
            language_token_features=language_tokens,
            language_token_mask=language_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            relation_weight=0.0,
        )
        loss_output.loss.backward()
        optimizer.step()
        losses.append(float(loss_output.loss.detach()))

    assert losses[-1] < losses[0] * 0.4
