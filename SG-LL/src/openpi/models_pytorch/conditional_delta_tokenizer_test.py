import torch

from openpi.models_pytorch.conditional_delta_tokenizer import ConditionalDeltaTokenizer
from openpi.models_pytorch.conditional_delta_tokenizer import ConditionalDeltaTokenizerConfig
from openpi.models_pytorch.conditional_delta_tokenizer import conditional_delta_tokenizer_loss


def _config(*, zero_init_decoder: bool = True) -> ConditionalDeltaTokenizerConfig:
    return ConditionalDeltaTokenizerConfig(
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
        max_tokens=8,
        max_horizon=16,
        relation_max_tokens=4,
        zero_init_decoder=zero_init_decoder,
    )


def _inputs():
    torch.manual_seed(0)
    current = torch.randn(2, 5, 12)
    future = current + 0.1 * torch.randn(2, 5, 12)
    language = torch.randn(2, 7)
    actions = torch.randn(2, 4, 3)
    state = torch.randn(2, 5)
    token_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    horizon_ids = torch.tensor([3, 4], dtype=torch.long)
    return current, future, language, actions, state, token_mask, horizon_ids


def test_conditional_delta_tokenizer_shapes() -> None:
    tokenizer = ConditionalDeltaTokenizer(_config())
    current, future, language, actions, state, token_mask, horizon_ids = _inputs()

    output = tokenizer(
        current,
        future,
        language_features=language,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )

    assert output.delta_tokens.shape == (2, 2, 6)
    assert output.future_recon.shape == current.shape


def test_conditional_delta_tokenizer_rejects_bad_condition_shapes() -> None:
    tokenizer = ConditionalDeltaTokenizer(_config())
    current, future, language, actions, state, token_mask, horizon_ids = _inputs()

    try:
        tokenizer(
            current,
            future,
            language_features=language[:, :-1],
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "language_features" in str(exc)
    else:
        raise AssertionError("bad language_features shape was accepted")

    try:
        tokenizer(
            current,
            future,
            language_features=language,
            actions=actions[:, :, :-1],
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "actions" in str(exc)
    else:
        raise AssertionError("bad actions shape was accepted")

    try:
        tokenizer(
            current,
            future,
            language_features=language,
            actions=actions,
            state=state[:, :-1],
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
    except ValueError as exc:
        assert "state" in str(exc)
    else:
        raise AssertionError("bad state shape was accepted")


def test_conditional_delta_zero_delta_identity_with_zero_init_decoder() -> None:
    tokenizer = ConditionalDeltaTokenizer(_config(zero_init_decoder=True))
    current, _future, language, actions, state, token_mask, horizon_ids = _inputs()
    zero_delta = torch.zeros(2, 2, 6)

    recon = tokenizer.decode(
        current,
        zero_delta,
        language_features=language,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )

    assert torch.allclose(recon, current.float(), atol=0.0, rtol=0.0)


def test_conditional_delta_tokenizer_gradients_reach_conditions() -> None:
    tokenizer = ConditionalDeltaTokenizer(_config())
    current, future, language, actions, state, token_mask, horizon_ids = _inputs()
    language.requires_grad_(True)
    actions.requires_grad_(True)
    state.requires_grad_(True)

    output = tokenizer(
        current,
        future,
        language_features=language,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )
    loss = output.delta_tokens.square().mean() + output.future_recon.square().mean()
    loss.backward()

    assert language.grad is not None and language.grad.abs().sum() > 0
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0


def test_conditional_delta_tokenizer_deterministic_eval() -> None:
    tokenizer = ConditionalDeltaTokenizer(_config()).eval()
    current, future, language, actions, state, token_mask, horizon_ids = _inputs()

    with torch.no_grad():
        output_a = tokenizer(
            current,
            future,
            language_features=language,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
        output_b = tokenizer(
            current,
            future,
            language_features=language,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )

    assert torch.allclose(output_a.delta_tokens, output_b.delta_tokens)
    assert torch.allclose(output_a.future_recon, output_b.future_recon)


def test_conditional_delta_tokenizer_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    tokenizer = ConditionalDeltaTokenizer(_config(zero_init_decoder=False))
    current, future, language, actions, state, token_mask, horizon_ids = _inputs()
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss_output = conditional_delta_tokenizer_loss(
            tokenizer,
            current,
            future,
            language_features=language,
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
