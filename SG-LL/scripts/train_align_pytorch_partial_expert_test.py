"""Focused regression tests for the partial action-expert ablation contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch.scheme4_adaptive_lookahead import Scheme4UtilityRouter
from openpi.models_pytorch.scheme4_adaptive_lookahead import utility_supervision_loss


def _load_trainer_module():
    script_path = Path(__file__).with_name("train_align_pytorch.py")
    spec = importlib.util.spec_from_file_location("_partial_expert_train_align", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER = _load_trainer_module()


class _ToyExpertCore(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(depth)])
        self.norm = nn.LayerNorm(4)


class _ToyExpert(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.model = _ToyExpertCore(depth)
        self.lm_head = nn.Linear(4, 4)


class _ToyPaliGemmaWithExpert(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.paligemma = nn.Linear(4, 4)
        self.gemma_expert = _ToyExpert(depth)


class _ToyPolicy(nn.Module):
    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.paligemma_with_expert = _ToyPaliGemmaWithExpert(depth)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        self.time_mlp_in = nn.Linear(4, 4)
        self.time_mlp_out = nn.Linear(4, 4)
        self.state_proj = nn.Linear(4, 4)
        self.action_time_mlp_in = nn.Linear(4, 4)
        self.action_time_mlp_out = nn.Linear(4, 4)
        self.planning = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.paligemma_with_expert.paligemma(inputs)
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            hidden = torch.tanh(layer(hidden))
        return self.planning(hidden)


class _ToyAdaptivePolicy(_ToyPolicy):
    def __init__(self, depth: int = 4) -> None:
        super().__init__(depth)
        self.spatial_adapter = nn.Linear(4, 4)
        self.future_predictor = nn.Linear(4, 4)
        self.scheme4_utility_router = Scheme4UtilityRouter(
            spatial_dim=4,
            language_dim=4,
            state_dim=4,
            action_intent_dim=4,
            hidden_dim=8,
        )
        self.scheme4_router_only_training = True


def _config(**overrides):
    values = {
        "freeze_vggt": True,
        "freeze_vlm_backbone": True,
        "train_action_expert": False,
        "partial_action_expert_last_n_blocks": 2,
        "partial_action_expert_lr": 5e-6,
        "lr_schedule": SimpleNamespace(peak_lr=5e-5),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _router_only_config(**overrides):
    values = {
        "freeze_vggt": True,
        "freeze_vlm_backbone": True,
        "train_action_expert": False,
        "partial_action_expert_last_n_blocks": 0,
        "partial_action_expert_lr": None,
        "scheme4_adaptive_lookahead_enabled": True,
        "scheme4_router_only_training": True,
        "lr_schedule": SimpleNamespace(peak_lr=5e-5),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _group_names(groups) -> dict[str, set[int]]:
    return {group.name: {id(param) for param in group.params} for group in groups}


def test_partial_expert_selects_only_the_final_two_gemma_blocks() -> None:
    model = _ToyPolicy(depth=4)
    vggt = nn.Linear(4, 4)

    groups = TRAINER.configure_trainable_parameters(model, vggt, _config(), is_main=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert {group.name for group in groups} == {"planning", "action_expert_tail"}
    assert all(not param.requires_grad for param in vggt.parameters())
    assert all(not param.requires_grad for param in model.paligemma_with_expert.paligemma.parameters())
    assert all(not param.requires_grad for param in model.action_out_proj.parameters())
    assert any(name.startswith("paligemma_with_expert.gemma_expert.model.layers.2.") for name in trainable)
    assert any(name.startswith("paligemma_with_expert.gemma_expert.model.layers.3.") for name in trainable)
    assert not any(name.startswith("paligemma_with_expert.gemma_expert.model.layers.0.") for name in trainable)
    assert not any(name.startswith("paligemma_with_expert.gemma_expert.model.layers.1.") for name in trainable)
    assert not any(
        name.startswith("paligemma_with_expert.gemma_expert.")
        and ".model.layers.2." not in name
        and ".model.layers.3." not in name
        for name in trainable
    )

    group_param_ids = _group_names(groups)
    assert group_param_ids["planning"].isdisjoint(group_param_ids["action_expert_tail"])
    assert group_param_ids["action_expert_tail"] == {
        id(param)
        for name, param in model.named_parameters()
        if name.startswith(
            (
                "paligemma_with_expert.gemma_expert.model.layers.2.",
                "paligemma_with_expert.gemma_expert.model.layers.3.",
            )
        )
    }


def test_default_frozen_expert_path_preserves_one_optimizer_group() -> None:
    model = _ToyPolicy(depth=4)
    vggt = nn.Linear(4, 4)

    groups = TRAINER.configure_trainable_parameters(
        model,
        vggt,
        _config(partial_action_expert_last_n_blocks=0, partial_action_expert_lr=None),
        is_main=False,
    )

    assert len(groups) == 1
    assert groups[0].name == "default"
    assert groups[0].lr_scale == 1.0
    assert all(not param.requires_grad for param in model.paligemma_with_expert.gemma_expert.parameters())
    assert all(not param.requires_grad for param in model.action_in_proj.parameters())
    assert all(not param.requires_grad for param in model.action_out_proj.parameters())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_action_expert": True}, "cannot combine"),
        ({"freeze_vlm_backbone": False}, "requires freeze_vlm_backbone"),
        ({"freeze_vggt": False}, "requires freeze_vggt"),
        ({"partial_action_expert_lr": None}, "requires partial_action_expert_lr"),
        ({"partial_action_expert_last_n_blocks": 5}, "exceeds action-expert depth"),
    ],
)
def test_partial_expert_rejects_ambiguous_or_invalid_trainable_sets(overrides, message: str) -> None:
    model = _ToyPolicy(depth=4)
    with pytest.raises(ValueError, match=message):
        TRAINER.configure_trainable_parameters(model, nn.Linear(4, 4), _config(**overrides), is_main=False)


def test_partial_expert_gradient_flow_stops_at_the_declared_boundary() -> None:
    torch.manual_seed(7)
    model = _ToyPolicy(depth=4)
    groups = TRAINER.configure_trainable_parameters(model, nn.Linear(4, 4), _config(), is_main=False)
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()

    selected_tail_params = {
        id(param) for param in next(group for group in groups if group.name == "action_expert_tail").params
    }
    for name, param in model.named_parameters():
        if id(param) in selected_tail_params:
            assert param.grad is not None, name
        elif name.startswith(("paligemma_with_expert.gemma_expert.", "paligemma_with_expert.paligemma.")):
            assert param.grad is None, name
    assert all(param.grad is not None for param in next(group for group in groups if group.name == "planning").params)


def test_scheduler_preserves_the_declared_ten_to_one_learning_rate_ratio() -> None:
    model = _ToyPolicy(depth=4)
    groups = TRAINER.configure_trainable_parameters(model, nn.Linear(4, 4), _config(), is_main=False)
    optimizer_groups = TRAINER.build_optimizer_parameter_groups(groups, peak_lr=5e-5)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=5e-5)

    TRAINER.set_scheduled_learning_rates(optimizer, 2.5e-5)

    by_name = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert by_name["planning"] == pytest.approx(2.5e-5)
    assert by_name["action_expert_tail"] == pytest.approx(2.5e-6)
    assert by_name["planning"] / by_name["action_expert_tail"] == pytest.approx(10.0)


def test_partial_expert_selection_is_deterministic() -> None:
    first = _ToyPolicy(depth=4)
    second = _ToyPolicy(depth=4)
    TRAINER.configure_trainable_parameters(first, nn.Linear(4, 4), _config(), is_main=False)
    TRAINER.configure_trainable_parameters(second, nn.Linear(4, 4), _config(), is_main=False)

    first_names = [name for name, param in first.named_parameters() if param.requires_grad]
    second_names = [name for name, param in second.named_parameters() if param.requires_grad]
    assert first_names == second_names


def test_router_only_freeze_and_gradient_boundary():
    torch.manual_seed(11)
    model = _ToyAdaptivePolicy()
    vggt = nn.Linear(4, 4)

    groups = TRAINER.configure_trainable_parameters(model, vggt, _router_only_config(), is_main=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {f"scheme4_utility_router.{name}" for name, _ in model.scheme4_utility_router.named_parameters()}

    assert {group.name for group in groups} == {"lookahead_router"}
    assert trainable == expected
    assert all(not parameter.requires_grad for parameter in vggt.parameters())

    router_inputs = {
        "spatial_tokens": torch.randn(3, 5, 4),
        "token_mask": torch.ones(3, 5, dtype=torch.bool),
        "language_features": torch.randn(3, 4),
        "state": torch.randn(3, 4),
        "action_intent": torch.randn(3, 4),
    }
    probability = model.scheme4_utility_router(**router_inputs)
    loss = utility_supervision_loss(probability, torch.tensor([0.9, 0.1, 0.7]))
    loss.backward()

    for name, parameter in model.named_parameters():
        if name in expected:
            assert parameter.grad is not None, name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_router_only_requires_adaptive_router_and_rejects_partial_expert():
    model = _ToyAdaptivePolicy()

    with pytest.raises(ValueError, match="requires scheme4_adaptive_lookahead_enabled"):
        TRAINER.configure_trainable_parameters(
            model,
            nn.Linear(4, 4),
            _router_only_config(scheme4_adaptive_lookahead_enabled=False),
            is_main=False,
        )
    with pytest.raises(ValueError, match="cannot combine with partial"):
        TRAINER.configure_trainable_parameters(
            model,
            nn.Linear(4, 4),
            _router_only_config(partial_action_expert_last_n_blocks=1, partial_action_expert_lr=5e-6),
            is_main=False,
        )
