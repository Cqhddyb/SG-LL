import pytest
import torch

from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3D3DeltaSuffixFusion
from openpi.models_pytorch.pi0_proposal_variants_pytorch import select_scheme4_training_intent_suffix
from openpi.models_pytorch.scheme4_adaptive_lookahead import Scheme4UtilityRouter
from openpi.models_pytorch.scheme4_adaptive_lookahead import resolve_lookahead_scale
from openpi.models_pytorch.scheme4_adaptive_lookahead import training_fusion_scale
from openpi.models_pytorch.scheme4_adaptive_lookahead import utility_supervision_loss
from openpi.models_pytorch.scheme4_adaptive_lookahead import utility_targets_from_losses


def _router_inputs():
    torch.manual_seed(4)
    return {
        "spatial_tokens": torch.randn(3, 5, 8),
        "token_mask": torch.tensor([[True, True, False, False, False], [True] * 5, [False] * 5]),
        "language_features": torch.randn(3, 6),
        "state": torch.randn(3, 7),
        "action_intent": torch.randn(3, 4),
    }


def _router():
    return Scheme4UtilityRouter(
        spatial_dim=8,
        language_dim=6,
        state_dim=7,
        action_intent_dim=4,
        hidden_dim=12,
    )


def test_router_outputs_finite_probabilities_and_handles_empty_spatial_mask():
    router = _router()
    inputs = _router_inputs()
    probability = router(**inputs)

    assert probability.shape == (3,)
    assert torch.isfinite(probability).all()
    assert torch.all((probability >= 0.0) & (probability <= 1.0))


def test_router_rejects_invalid_contracts():
    router = _router()
    inputs = _router_inputs()

    with pytest.raises(ValueError, match="token_mask"):
        router(**{**inputs, "token_mask": inputs["token_mask"][:, :-1]})
    with pytest.raises(ValueError, match="state shape"):
        router(**{**inputs, "state": inputs["state"][:, :-1]})


def test_router_forward_backward_updates_all_condition_encoders():
    router = _router()
    probability = router(**_router_inputs())
    target = torch.tensor([0.9, 0.1, 0.7])
    loss = utility_supervision_loss(probability, target)
    loss.backward()

    for module in (router.spatial_encoder, router.language_encoder, router.state_encoder, router.action_intent_encoder):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in router.fusion.parameters()
    )


def test_utility_target_is_detached_and_matches_action_loss_benefit():
    base_loss = torch.tensor([2.0, 0.5, 1.0], requires_grad=True)
    lookahead_loss = torch.tensor([1.0, 1.5, 1.0], requires_grad=True)

    benefit, target = utility_targets_from_losses(
        base_loss,
        lookahead_loss,
        temperature=0.5,
    )

    expected_benefit = torch.tensor([1.0, -1.0, 0.0])
    assert torch.equal(benefit, expected_benefit)
    assert torch.allclose(target, torch.sigmoid(expected_benefit / 0.5))
    assert not benefit.requires_grad
    assert not target.requires_grad
    assert base_loss.grad is None
    assert lookahead_loss.grad is None


def test_utility_targets_respect_margin_and_reject_bad_values():
    benefit, target = utility_targets_from_losses(
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        temperature=0.1,
        margin=0.5,
    )
    assert benefit.item() == pytest.approx(0.5)
    assert target.item() == pytest.approx(0.5)

    with pytest.raises(ValueError, match="positive"):
        utility_targets_from_losses(torch.ones(1), torch.zeros(1), temperature=0.0)
    with pytest.raises(ValueError, match="must match"):
        utility_targets_from_losses(torch.ones(2), torch.zeros(1), temperature=1.0)


def test_routing_modes_return_expected_no_fixed_soft_and_hard_scales():
    probability = torch.tensor([0.2, 0.5, 0.8])

    assert torch.equal(resolve_lookahead_scale(probability, "none"), torch.zeros_like(probability))
    assert torch.equal(resolve_lookahead_scale(probability, "fixed"), torch.ones_like(probability))
    assert torch.equal(resolve_lookahead_scale(probability, "soft"), probability)
    assert torch.equal(resolve_lookahead_scale(probability, "hard"), torch.tensor([0.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="mode"):
        resolve_lookahead_scale(probability, "oracle")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_lookahead_scale(torch.tensor([1.1]), "soft")


def test_per_example_fusion_scale_gates_future_residual_independently():
    fusion = Scheme3D3DeltaSuffixFusion(suffix_dim=12, delta_dim=6, num_heads=3, dropout=0.0)
    fusion.gate.data.fill_(0.1)
    suffix = torch.randn(2, 4, 12)
    delta = torch.randn(2, 3, 6)
    scale = torch.tensor([0.0, 1.0], requires_grad=True)

    fused = fusion(suffix, delta, residual_scale=scale)
    residual = fused - suffix
    assert torch.equal(residual[0], torch.zeros_like(residual[0]))
    assert residual[1].abs().sum() > 0

    fused.square().mean().backward()
    assert scale.grad is not None
    assert scale.grad[1].abs() > 0


def test_zero_lookahead_scale_is_exact_identity_even_when_future_gate_is_open():
    fusion = Scheme3D3DeltaSuffixFusion(suffix_dim=12, delta_dim=6, num_heads=3, dropout=0.0)
    fusion.gate.data.fill_(0.3)
    suffix = torch.randn(2, 4, 12)
    delta = torch.randn(2, 3, 6)

    no_look = fusion(suffix, delta, residual_scale=resolve_lookahead_scale(torch.tensor([0.2, 0.8]), "none"))
    fixed = fusion(suffix, delta, residual_scale=resolve_lookahead_scale(torch.tensor([0.2, 0.8]), "fixed"))

    assert torch.equal(no_look, suffix.float())
    assert not torch.equal(fixed, suffix.float())


def test_training_intent_selects_exact_deployment_t1_suffix_when_requested():
    sampled = torch.randn(2, 4, 6, requires_grad=True)
    initial_noise = torch.randn(2, 4, 6, requires_grad=True)

    legacy = select_scheme4_training_intent_suffix(sampled, None, "sampled_flow")
    aligned = select_scheme4_training_intent_suffix(sampled, initial_noise, "t1_noise")

    assert legacy is sampled
    assert aligned is initial_noise
    assert aligned.shape == sampled.shape
    assert torch.equal(aligned, initial_noise)
    with pytest.raises(ValueError, match="required for t1_noise"):
        select_scheme4_training_intent_suffix(sampled, None, "t1_noise")
    with pytest.raises(ValueError, match="must match"):
        select_scheme4_training_intent_suffix(sampled, torch.randn(1, 4, 6), "t1_noise")
    with pytest.raises(ValueError, match="must be"):
        select_scheme4_training_intent_suffix(sampled, None, "ground_truth_action")


def test_router_only_flow_fusion_does_not_train_router_outside_utility_loss():
    probability = torch.tensor([0.2, 0.8], requires_grad=True)
    scale = training_fusion_scale(probability, router_only=True)
    assert torch.equal(scale, probability.detach())
    assert not scale.requires_grad

    frozen_residual = torch.tensor([3.0, -2.0], requires_grad=True)
    action_loss = (scale * frozen_residual).sum()
    action_loss.backward()
    assert probability.grad is None

    utility_probability = torch.tensor([0.2, 0.8], requires_grad=True)
    utility_loss = utility_supervision_loss(utility_probability, torch.tensor([0.9, 0.1]))
    utility_loss.backward()
    assert utility_probability.grad is not None
    assert utility_probability.grad.abs().sum() > 0
