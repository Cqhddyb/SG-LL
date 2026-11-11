import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3CorrectedDeltaAdapter
from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3DeltaQueryPredictor
from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3DeployableCurrentAdapter


def _current_adapter() -> Scheme3DeployableCurrentAdapter:
    return Scheme3DeployableCurrentAdapter(
        visual_dim=12,
        teacher_dim=10,
        hidden_dim=24,
        dropout=0.0,
    )


def _predictor() -> Scheme3DeltaQueryPredictor:
    return Scheme3DeltaQueryPredictor(
        source_dim=10,
        language_dim=12,
        state_dim=7,
        hidden_dim=24,
        delta_dim=8,
        num_delta_tokens=3,
        num_layers=2,
        num_heads=4,
        max_horizon=16,
        horizon_id=4,
        dropout=0.0,
    )


def _intent_predictor() -> Scheme3DeltaQueryPredictor:
    return Scheme3DeltaQueryPredictor(
        source_dim=10,
        language_dim=12,
        state_dim=7,
        hidden_dim=24,
        delta_dim=8,
        num_delta_tokens=3,
        num_layers=2,
        num_heads=4,
        max_horizon=16,
        horizon_id=4,
        action_intent_dim=6,
        dropout=0.0,
    )


def _adapter() -> Scheme3CorrectedDeltaAdapter:
    return Scheme3CorrectedDeltaAdapter(
        source_dim=10,
        language_dim=12,
        state_dim=7,
        suffix_dim=16,
        hidden_dim=24,
        delta_dim=8,
        num_delta_tokens=3,
        num_layers=2,
        num_heads=4,
        max_horizon=16,
        horizon_id=4,
        dropout=0.0,
    )


def _intent_adapter() -> Scheme3CorrectedDeltaAdapter:
    return Scheme3CorrectedDeltaAdapter(
        source_dim=10,
        language_dim=12,
        state_dim=7,
        suffix_dim=16,
        hidden_dim=24,
        delta_dim=8,
        num_delta_tokens=3,
        num_layers=2,
        num_heads=4,
        max_horizon=16,
        horizon_id=4,
        action_intent_dim=6,
        dropout=0.0,
    )


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(0)
    source_tokens = torch.randn(2, 5, 10)
    suffix_out = torch.randn(2, 4, 16)
    language_features = torch.randn(2, 12)
    state = torch.randn(2, 7)
    token_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    vision_hidden = torch.randn(2, 5, 12)
    action_intent = torch.randn(2, 6)
    return source_tokens, suffix_out, language_features, state, token_mask, vision_hidden, action_intent


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_deployable_current_adapter_shapes_masks_and_gradients() -> None:
    adapter = _current_adapter()
    _source_tokens, _suffix_out, _language_features, _state, token_mask, vision_hidden, _action_intent = _inputs()
    vision_hidden = vision_hidden.requires_grad_()

    current_tokens = adapter(vision_hidden, token_mask)
    loss = current_tokens.square().mean()
    loss.backward()

    assert current_tokens.shape == (2, 5, 10)
    assert torch.equal(current_tokens[0, 3:], torch.zeros_like(current_tokens[0, 3:]))
    assert torch.equal(current_tokens[1, 4:], torch.zeros_like(current_tokens[1, 4:]))
    assert vision_hidden.grad is not None
    assert torch.isfinite(vision_hidden.grad).all()
    assert _has_nonzero_grad(adapter)


def test_deployable_current_adapter_rejects_bad_shapes() -> None:
    adapter = _current_adapter()
    _source_tokens, _suffix_out, _language_features, _state, token_mask, vision_hidden, _action_intent = _inputs()

    with pytest.raises(ValueError, match="vision_hidden"):
        adapter(vision_hidden[:, :, :-1], token_mask)

    with pytest.raises(ValueError, match="token_mask"):
        adapter(vision_hidden, token_mask[:, :-1])


def test_delta_query_predictor_shapes_and_bad_shapes() -> None:
    predictor = _predictor()
    source_tokens, _suffix_out, language_features, state, token_mask, _vision_hidden, action_intent = _inputs()

    pred_delta = predictor(
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
    )

    assert pred_delta.shape == (2, 3, 8)

    with pytest.raises(ValueError, match="source_tokens"):
        predictor(source_tokens[:, :, :-1], token_mask=token_mask, language_features=language_features, state=state)

    with pytest.raises(ValueError, match="token_mask"):
        predictor(source_tokens, token_mask=token_mask[:, :-1], language_features=language_features, state=state)

    with pytest.raises(ValueError, match="language_features"):
        predictor(source_tokens, token_mask=token_mask, language_features=language_features[:, :-1], state=state)

    with pytest.raises(ValueError, match="state"):
        predictor(source_tokens, token_mask=token_mask, language_features=language_features, state=state[:, :-1])

    with pytest.raises(ValueError, match="action_intent"):
        predictor(
            source_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )


def test_delta_query_predictor_action_intent_condition_shapes_and_bad_shapes() -> None:
    predictor = _intent_predictor()
    source_tokens, _suffix_out, language_features, state, token_mask, _vision_hidden, action_intent = _inputs()

    pred_delta = predictor(
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )

    assert pred_delta.shape == (2, 3, 8)

    with pytest.raises(ValueError, match="action_intent is required"):
        predictor(source_tokens, token_mask=token_mask, language_features=language_features, state=state)

    with pytest.raises(ValueError, match="action_intent shape"):
        predictor(
            source_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent[:, :-1],
        )


def test_corrected_delta_adapter_shapes_and_zero_gate_identity() -> None:
    adapter = _adapter()
    source_tokens, suffix_out, language_features, state, token_mask, _vision_hidden, _action_intent = _inputs()

    refined, pred_delta = adapter(
        suffix_out,
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
    )

    assert refined.shape == suffix_out.shape
    assert pred_delta.shape == (2, 3, 8)
    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)
    assert float(adapter.gate_abs_mean().detach()) == 0.0
    assert float(adapter.gate_max_abs().detach()) == 0.0


def test_corrected_delta_adapter_zero_gate_gradient_reaches_gate() -> None:
    adapter = _adapter()
    source_tokens, suffix_out, language_features, state, token_mask, _vision_hidden, _action_intent = _inputs()

    refined, _pred_delta = adapter(
        suffix_out,
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
    )
    loss = refined.square().mean()
    loss.backward()

    gate_grad = adapter.suffix_fusion.gate.grad
    assert gate_grad is not None
    assert torch.isfinite(gate_grad).all()
    assert gate_grad.abs().sum() > 0


def test_corrected_delta_adapter_open_gate_gradient_reaches_predictor_and_fusion() -> None:
    adapter = _adapter()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    source_tokens, suffix_out, language_features, state, token_mask, _vision_hidden, _action_intent = _inputs()

    refined, pred_delta = adapter(
        suffix_out,
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
    )
    loss = refined.square().mean() + 0.1 * pred_delta.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.predictor)
    assert _has_nonzero_grad(adapter.suffix_fusion)


def test_corrected_delta_adapter_deterministic_eval() -> None:
    adapter = _adapter().eval()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    source_tokens, suffix_out, language_features, state, token_mask, _vision_hidden, _action_intent = _inputs()

    with torch.no_grad():
        refined_a, delta_a = adapter(
            suffix_out,
            source_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
        )
        refined_b, delta_b = adapter(
            suffix_out,
            source_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
        )

    assert torch.allclose(refined_a, refined_b)
    assert torch.allclose(delta_a, delta_b)


def test_delta_query_predictor_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    predictor = _predictor()
    source_tokens, _suffix_out, language_features, state, token_mask, _vision_hidden, _action_intent = _inputs()
    target_delta = torch.tanh(torch.randn(2, 3, 8))
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        pred_delta = predictor(
            source_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
        )
        loss = F.mse_loss(pred_delta, target_delta)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.4


def test_corrected_delta_intent_adapter_zero_gate_identity_and_gradient() -> None:
    adapter = _intent_adapter()
    source_tokens, suffix_out, language_features, state, token_mask, _vision_hidden, action_intent = _inputs()

    refined, pred_delta = adapter(
        suffix_out,
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )

    assert refined.shape == suffix_out.shape
    assert pred_delta.shape == (2, 3, 8)
    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)

    adapter.suffix_fusion.gate.data.fill_(0.1)
    refined, pred_delta = adapter(
        suffix_out,
        source_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )
    loss = refined.square().mean() + 0.1 * pred_delta.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.predictor)
    assert _has_nonzero_grad(adapter.suffix_fusion)
