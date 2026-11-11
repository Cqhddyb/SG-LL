import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme4LanguageActionDeltaAdapter
from openpi.models_pytorch.pi0_proposal_variants_pytorch import apply_scheme4_inference_condition_ablation
from openpi.models_pytorch.pi0_proposal_variants_pytorch import apply_scheme4_inference_delta_control
from openpi.models_pytorch.pi0_proposal_variants_pytorch import apply_scheme4_training_action_condition
from openpi.models_pytorch.scheme4_condition_diagnostics import delta_token_distance
from openpi.models_pytorch.scheme4_condition_diagnostics import margin_negative_condition_loss
from openpi.models_pytorch.scheme4_condition_diagnostics import roll_batch_condition


def _adapter(*, delta_enabled: bool = True) -> Scheme4LanguageActionDeltaAdapter:
    return Scheme4LanguageActionDeltaAdapter(
        spatial_dim=10,
        language_dim=12,
        state_dim=7,
        action_intent_dim=6,
        suffix_dim=16,
        hidden_dim=24,
        delta_dim=8,
        num_delta_tokens=3,
        num_layers=2,
        num_heads=4,
        max_horizon=16,
        horizon_id=4,
        action_intent_mode="t1_suffix_mean",
        delta_fusion_type="suffix_cross_attn",
        delta_enabled=delta_enabled,
        dropout=0.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    suffix_out = torch.randn(2, 4, 16)
    spatial_tokens = torch.randn(2, 5, 10)
    token_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    language_features = torch.randn(2, 12)
    state = torch.randn(2, 7)
    action_intent = torch.randn(2, 6)
    return suffix_out, spatial_tokens, token_mask, language_features, state, action_intent


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_scheme4_lac_adapter_shapes_and_zero_gate_identity() -> None:
    adapter = _adapter()
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    refined, pred_delta, delta_residual = adapter(
        suffix_out,
        spatial_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )

    assert refined.shape == suffix_out.shape
    assert pred_delta.shape == (2, 3, 8)
    assert delta_residual.shape == suffix_out.shape
    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)
    assert torch.allclose(delta_residual, torch.zeros_like(delta_residual), atol=0.0, rtol=0.0)
    assert float(adapter.gate_abs_mean().detach()) == 0.0
    assert float(adapter.gate_max_abs().detach()) == 0.0


def test_scheme4_lac_adapter_rejects_bad_shapes() -> None:
    adapter = _adapter()
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    with pytest.raises(ValueError, match="suffix_out"):
        adapter(
            suffix_out[:, :, :-1],
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    with pytest.raises(ValueError, match="spatial_tokens"):
        adapter(
            suffix_out,
            spatial_tokens[:, :, :-1],
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    with pytest.raises(ValueError, match="token_mask"):
        adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask[:, :-1],
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    with pytest.raises(ValueError, match="language_features"):
        adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features[:, :-1],
            state=state,
            action_intent=action_intent,
        )

    with pytest.raises(ValueError, match="state"):
        adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state[:, :-1],
            action_intent=action_intent,
        )

    with pytest.raises(ValueError, match="action_intent"):
        adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent[:, :-1],
        )


def test_scheme4_lac_adapter_zero_gate_gradient_reaches_gate() -> None:
    adapter = _adapter()
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    refined, _pred_delta, _delta_residual = adapter(
        suffix_out,
        spatial_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )
    loss = refined.square().mean()
    loss.backward()

    gate_grad = adapter.suffix_fusion.gate.grad
    assert gate_grad is not None
    assert torch.isfinite(gate_grad).all()
    assert gate_grad.abs().sum() > 0


def test_scheme4_lac_adapter_open_gate_gradient_reaches_predictor_and_fusion() -> None:
    adapter = _adapter()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    refined, pred_delta, _delta_residual = adapter(
        suffix_out,
        spatial_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )
    loss = refined.square().mean() + 0.1 * pred_delta.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.predictor)
    assert _has_nonzero_grad(adapter.suffix_fusion)


def test_scheme4_lac_adapter_inference_fusion_scale_is_linear_and_zero_is_identity() -> None:
    adapter = _adapter().eval()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    with torch.no_grad():
        pred_delta = adapter.predict_delta(
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        zero = adapter.suffix_fusion(suffix_out, pred_delta, residual_scale=0.0)
        half = adapter.suffix_fusion(suffix_out, pred_delta, residual_scale=0.5)
        full = adapter.suffix_fusion(suffix_out, pred_delta, residual_scale=1.0)

    assert torch.allclose(zero, suffix_out.float(), atol=0.0, rtol=0.0)
    assert torch.allclose(half - suffix_out.float(), 0.5 * (full - suffix_out.float()), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="residual_scale"):
        adapter.suffix_fusion(suffix_out, pred_delta, residual_scale=-0.1)


def test_scheme4_lac_adapter_delta_disabled_identity_and_frozen_params() -> None:
    adapter = _adapter(delta_enabled=False)
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    refined, pred_delta, delta_residual = adapter(
        suffix_out,
        spatial_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )

    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)
    assert torch.allclose(pred_delta, torch.zeros_like(pred_delta), atol=0.0, rtol=0.0)
    assert torch.allclose(delta_residual, torch.zeros_like(delta_residual), atol=0.0, rtol=0.0)
    assert not any(param.requires_grad for param in adapter.parameters())


def test_scheme4_lac_adapter_deterministic_eval() -> None:
    adapter = _adapter().eval()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()

    with torch.no_grad():
        refined_a, delta_a, residual_a = adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        refined_b, delta_b, residual_b = adapter(
            suffix_out,
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    assert torch.allclose(refined_a, refined_b)
    assert torch.allclose(delta_a, delta_b)
    assert torch.allclose(residual_a, residual_b)


def test_scheme4_inference_condition_ablation_modes() -> None:
    _suffix_out, _spatial_tokens, _token_mask, language_features, _state, action_intent = _inputs()

    lang_none, action_none = apply_scheme4_inference_condition_ablation(
        "none",
        language_features,
        action_intent,
    )
    assert lang_none is language_features
    assert action_none is action_intent

    lang_zero, action_same = apply_scheme4_inference_condition_ablation(
        "zero_language",
        language_features,
        action_intent,
    )
    assert torch.allclose(lang_zero, torch.zeros_like(language_features), atol=0.0, rtol=0.0)
    assert action_same is action_intent

    lang_same, action_zero = apply_scheme4_inference_condition_ablation(
        "zero_action",
        language_features,
        action_intent,
    )
    assert lang_same is language_features
    assert torch.allclose(action_zero, torch.zeros_like(action_intent), atol=0.0, rtol=0.0)

    lang_zero_both, action_zero_both = apply_scheme4_inference_condition_ablation(
        "zero_language_action",
        language_features,
        action_intent,
    )
    assert torch.allclose(lang_zero_both, torch.zeros_like(language_features), atol=0.0, rtol=0.0)
    assert torch.allclose(action_zero_both, torch.zeros_like(action_intent), atol=0.0, rtol=0.0)

    previous_language = language_features + 10.0
    previous_action = action_intent - 10.0
    wrong_language, action_for_wrong_language = apply_scheme4_inference_condition_ablation(
        "wrong_language",
        language_features,
        action_intent,
        previous_language_features=previous_language,
        previous_action_intent=previous_action,
    )
    assert torch.allclose(wrong_language, previous_language, atol=0.0, rtol=0.0)
    assert action_for_wrong_language is action_intent

    language_for_wrong_action, wrong_action = apply_scheme4_inference_condition_ablation(
        "wrong_action",
        language_features,
        action_intent,
        previous_language_features=previous_language,
        previous_action_intent=previous_action,
    )
    assert language_for_wrong_action is language_features
    assert torch.allclose(wrong_action, previous_action, atol=0.0, rtol=0.0)

    shuffled_language, action_for_shuffled_language = apply_scheme4_inference_condition_ablation(
        "shuffled_language",
        language_features,
        action_intent,
    )
    assert torch.allclose(shuffled_language, torch.roll(language_features, shifts=1, dims=0), atol=0.0, rtol=0.0)
    assert action_for_shuffled_language is action_intent

    language_for_shuffled_action, shuffled_action = apply_scheme4_inference_condition_ablation(
        "shuffled_action",
        language_features,
        action_intent,
    )
    assert language_for_shuffled_action is language_features
    assert torch.allclose(shuffled_action, torch.roll(action_intent, shifts=1, dims=0), atol=0.0, rtol=0.0)

    singleton_language_input = language_features[:1]
    singleton_action_input = action_intent[:1]
    singleton_language, singleton_action = apply_scheme4_inference_condition_ablation(
        "shuffled_language",
        singleton_language_input,
        singleton_action_input,
    )
    assert torch.allclose(singleton_language, torch.zeros_like(singleton_language_input), atol=0.0, rtol=0.0)
    assert singleton_action is singleton_action_input


def test_scheme4_inference_condition_ablation_rejects_bad_inputs() -> None:
    _suffix_out, _spatial_tokens, _token_mask, language_features, _state, action_intent = _inputs()

    with pytest.raises(ValueError, match="scheme4_inference_condition_ablation"):
        apply_scheme4_inference_condition_ablation("bad_mode", language_features, action_intent)
    with pytest.raises(ValueError, match="language_features"):
        apply_scheme4_inference_condition_ablation("none", language_features.unsqueeze(0), action_intent)
    with pytest.raises(ValueError, match="action_intent"):
        apply_scheme4_inference_condition_ablation("none", language_features, action_intent.unsqueeze(0))
    with pytest.raises(ValueError, match="same batch size"):
        apply_scheme4_inference_condition_ablation("none", language_features[:1], action_intent)


def test_scheme4_training_action_condition_is_explicit_and_shape_preserving() -> None:
    _suffix_out, _spatial_tokens, _token_mask, _language_features, _state, action_intent = _inputs()

    normal = apply_scheme4_training_action_condition("normal", action_intent)
    zero = apply_scheme4_training_action_condition("zero_action", action_intent)

    assert normal is action_intent
    assert zero.shape == action_intent.shape
    assert zero.dtype == action_intent.dtype
    assert zero.device == action_intent.device
    assert torch.allclose(zero, torch.zeros_like(action_intent), atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="scheme4_training_action_condition"):
        apply_scheme4_training_action_condition("wrong", action_intent)
    with pytest.raises(ValueError, match="rank 2"):
        apply_scheme4_training_action_condition("zero_action", action_intent.unsqueeze(0))


def test_scheme4_inference_delta_control_modes() -> None:
    torch.manual_seed(2)
    pred_delta = torch.randn(2, 3, 8)
    previous_delta = pred_delta + 5.0
    mean_delta = torch.arange(24, dtype=torch.float32).view(3, 8)

    normal = apply_scheme4_inference_delta_control("normal", pred_delta)
    assert normal is pred_delta

    zero = apply_scheme4_inference_delta_control("zero_delta", pred_delta)
    assert zero.shape == pred_delta.shape
    assert torch.allclose(zero, torch.zeros_like(pred_delta), atol=0.0, rtol=0.0)

    mean = apply_scheme4_inference_delta_control("mean_delta", pred_delta, mean_delta_tokens=mean_delta)
    assert mean.shape == pred_delta.shape
    assert torch.allclose(mean[0], mean_delta, atol=0.0, rtol=0.0)
    assert torch.allclose(mean[1], mean_delta, atol=0.0, rtol=0.0)

    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    random_a = apply_scheme4_inference_delta_control("random_delta", pred_delta, generator=generator_a)
    random_b = apply_scheme4_inference_delta_control("random_delta", pred_delta, generator=generator_b)
    assert random_a.shape == pred_delta.shape
    assert torch.allclose(random_a, random_b, atol=0.0, rtol=0.0)
    assert not torch.allclose(random_a, pred_delta)

    generator_c = torch.Generator().manual_seed(11)
    norm_matched = apply_scheme4_inference_delta_control(
        "norm_matched_random_delta",
        pred_delta,
        generator=generator_c,
    )
    assert norm_matched.shape == pred_delta.shape
    assert torch.allclose(
        norm_matched.float().norm(dim=-1),
        pred_delta.float().norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )

    cross_episode = apply_scheme4_inference_delta_control(
        "cross_episode_delta",
        pred_delta,
        previous_delta_tokens=previous_delta,
    )
    assert torch.allclose(cross_episode, previous_delta, atol=0.0, rtol=0.0)

    batch_shuffled = apply_scheme4_inference_delta_control("batch_shuffled_delta", pred_delta)
    assert torch.allclose(batch_shuffled, torch.roll(pred_delta, shifts=1, dims=0), atol=0.0, rtol=0.0)

    singleton_shuffled = apply_scheme4_inference_delta_control(
        "batch_shuffled_delta",
        pred_delta[:1],
        previous_delta_tokens=previous_delta[:1],
    )
    assert torch.allclose(singleton_shuffled, previous_delta[:1], atol=0.0, rtol=0.0)


def test_scheme4_inference_delta_control_rejects_bad_inputs() -> None:
    pred_delta = torch.randn(2, 3, 8)
    with pytest.raises(ValueError, match="scheme4_inference_delta_control"):
        apply_scheme4_inference_delta_control("bad_mode", pred_delta)
    with pytest.raises(ValueError, match="pred_delta_tokens"):
        apply_scheme4_inference_delta_control("normal", pred_delta.unsqueeze(0))
    with pytest.raises(ValueError, match="requires"):
        apply_scheme4_inference_delta_control("mean_delta", pred_delta)
    with pytest.raises(ValueError, match="mean_delta_tokens shape"):
        apply_scheme4_inference_delta_control("mean_delta", pred_delta, mean_delta_tokens=torch.randn(4, 8))


def test_scheme4_lac_delta_predictor_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    adapter = _adapter()
    _suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()
    target_delta = torch.tanh(torch.randn(2, 3, 8))
    optimizer = torch.optim.AdamW(adapter.predictor.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        pred_delta = adapter.predict_delta(
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        loss = F.mse_loss(pred_delta, target_delta)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.4


def test_scheme4_lac_condition_negative_loss_reaches_predictor() -> None:
    adapter = _adapter()
    _suffix_out, spatial_tokens, token_mask, language_features, state, action_intent = _inputs()
    target_delta = torch.randn(2, 3, 8)

    pred_delta = adapter.predict_delta(
        spatial_tokens,
        token_mask=token_mask,
        language_features=language_features,
        state=state,
        action_intent=action_intent,
    )
    neg_language, valid = roll_batch_condition(language_features)
    neg_delta = adapter.predict_delta(
        spatial_tokens,
        token_mask=token_mask,
        language_features=neg_language,
        state=state,
        action_intent=action_intent,
    )
    normal_distance = delta_token_distance(pred_delta, target_delta)
    negative_distance = delta_token_distance(neg_delta, target_delta)
    loss = margin_negative_condition_loss(normal_distance, negative_distance, margin=0.05, valid=valid)
    loss.backward()

    assert loss.ndim == 0
    assert _has_nonzero_grad(adapter.predictor)
