# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_MODULE_PATH = Path(__file__).with_name("serve_policy_proposal.py")
_SPEC = importlib.util.spec_from_file_location("serve_policy_proposal", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
serve_policy_proposal = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = serve_policy_proposal
_SPEC.loader.exec_module(serve_policy_proposal)


@dataclasses.dataclass(frozen=True)
class _FakeData:
    future_teacher_obs_offset: int | None = 10


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    data: _FakeData = dataclasses.field(default_factory=_FakeData)
    scheme3_horizon_id: int | None = None
    scheme4_delta_enabled: bool = True
    scheme4_delta_token_loss_coeff: float = 0.0
    scheme4_future_recon_loss_coeff: float = 0.01
    scheme4_training_action_condition: str = "normal"
    scheme4_condition_negative_modes: str = "language,action"
    scheme4_inference_condition_ablation: str = "none"
    scheme4_inference_delta_control: str = "normal"
    scheme4_inference_delta_mean_path: str | None = None
    scheme4_inference_control_seed: int = 0
    scheme4_inference_fusion_scale: float = 1.0
    scheme4_inference_diagnostics_enabled: bool = False
    scheme4_adaptive_lookahead_enabled: bool = False
    scheme4_inference_lookahead_mode: str = "fixed"
    scheme4_tokenizer_checkpoint: str | None = None


def _args(
    *,
    action_condition: str = "default",
    condition_negative_modes: str = "default",
    fusion_scale: float | None = None,
    diagnostics_enabled: str = "default",
    lookahead_mode: str = "default",
) -> SimpleNamespace:
    return SimpleNamespace(
        policy_data_future_teacher_obs_offset=None,
        policy_scheme3_horizon_id=None,
        policy_scheme4_delta_enabled="default",
        policy_scheme4_training_action_condition=action_condition,
        policy_scheme4_condition_negative_modes=condition_negative_modes,
        policy_scheme4_inference_condition_ablation="default",
        policy_scheme4_inference_delta_control="default",
        policy_scheme4_inference_delta_mean_path=None,
        policy_scheme4_inference_control_seed=None,
        policy_scheme4_inference_fusion_scale=fusion_scale,
        policy_scheme4_inference_diagnostics_enabled=diagnostics_enabled,
        policy_scheme4_lookahead_mode=lookahead_mode,
        policy_scheme4_tokenizer_checkpoint=None,
    )


def test_training_action_condition_override_is_explicit_and_default_is_identity() -> None:
    config = _FakeTrainConfig()

    assert serve_policy_proposal._apply_policy_config_overrides(config, _args()) is config
    overridden = serve_policy_proposal._apply_policy_config_overrides(config, _args(action_condition="zero_action"))
    assert overridden.scheme4_training_action_condition == "zero_action"
    assert overridden.scheme4_delta_enabled is True
    assert overridden.data is config.data


def test_condition_negative_modes_can_reconstruct_zero_action_checkpoint_config() -> None:
    config = _FakeTrainConfig()

    overridden = serve_policy_proposal._apply_policy_config_overrides(
        config,
        _args(action_condition="zero_action", condition_negative_modes="language"),
    )

    assert overridden.scheme4_training_action_condition == "zero_action"
    assert overridden.scheme4_condition_negative_modes == "language"
    assert config.scheme4_condition_negative_modes == "language,action"


def test_inference_fusion_scale_override_is_explicit_and_validated() -> None:
    config = _FakeTrainConfig()
    assert serve_policy_proposal._apply_policy_config_overrides(config, _args()) is config
    overridden = serve_policy_proposal._apply_policy_config_overrides(config, _args(fusion_scale=0.25))
    assert overridden.scheme4_inference_fusion_scale == 0.25
    with pytest.raises(ValueError, match="fusion-scale"):
        serve_policy_proposal._apply_policy_config_overrides(config, _args(fusion_scale=-0.1))


def test_inference_diagnostics_override_is_explicit_and_default_is_identity() -> None:
    config = _FakeTrainConfig()

    assert serve_policy_proposal._apply_policy_config_overrides(config, _args()) is config
    enabled = serve_policy_proposal._apply_policy_config_overrides(config, _args(diagnostics_enabled="true"))
    disabled = serve_policy_proposal._apply_policy_config_overrides(config, _args(diagnostics_enabled="false"))

    assert enabled.scheme4_inference_diagnostics_enabled is True
    assert disabled.scheme4_inference_diagnostics_enabled is False
    assert config.scheme4_inference_diagnostics_enabled is False


def test_lookahead_mode_override_is_explicit_and_soft_mode_enables_router_config() -> None:
    config = _FakeTrainConfig()

    assert serve_policy_proposal._apply_policy_config_overrides(config, _args()) is config
    no_look = serve_policy_proposal._apply_policy_config_overrides(config, _args(lookahead_mode="none"))
    soft = serve_policy_proposal._apply_policy_config_overrides(config, _args(lookahead_mode="soft"))

    assert no_look.scheme4_inference_lookahead_mode == "none"
    assert no_look.scheme4_adaptive_lookahead_enabled is False
    assert soft.scheme4_inference_lookahead_mode == "soft"
    assert soft.scheme4_adaptive_lookahead_enabled is True
    assert config.scheme4_inference_lookahead_mode == "fixed"


def test_checkpoint_metadata_restores_router_module_for_none_and_fixed_modes() -> None:
    checkpoint_config = {"scheme4_adaptive_lookahead_enabled": True}

    for mode in ("none", "fixed", "soft", "hard"):
        requested = serve_policy_proposal._apply_policy_config_overrides(
            _FakeTrainConfig(), _args(lookahead_mode=mode)
        )
        restored = serve_policy_proposal._apply_checkpoint_architecture(requested, checkpoint_config)
        assert restored.scheme4_adaptive_lookahead_enabled is True


def test_checkpoint_without_router_metadata_keeps_legacy_architecture() -> None:
    config = _FakeTrainConfig()

    restored = serve_policy_proposal._apply_checkpoint_architecture(
        config, {"scheme4_adaptive_lookahead_enabled": False}
    )

    assert restored.scheme4_adaptive_lookahead_enabled is False
    assert config.scheme4_adaptive_lookahead_enabled is False


def test_policy_rng_seed_is_optional_and_seeds_cuda_when_available() -> None:
    with mock.patch.object(serve_policy_proposal.torch, "manual_seed") as seed_cpu, mock.patch.object(
        serve_policy_proposal.torch.cuda, "is_available", return_value=True
    ), mock.patch.object(serve_policy_proposal.torch.cuda, "manual_seed_all") as seed_cuda:
        serve_policy_proposal.seed_policy_rng(None)
        seed_cpu.assert_not_called()
        seed_cuda.assert_not_called()

        serve_policy_proposal.seed_policy_rng(17)

    seed_cpu.assert_called_once_with(17)
    seed_cuda.assert_called_once_with(17)


def test_training_action_condition_override_rejects_missing_field() -> None:
    @dataclasses.dataclass(frozen=True)
    class MissingFieldConfig:
        data: _FakeData = dataclasses.field(default_factory=_FakeData)

    with pytest.raises(ValueError, match="scheme4_training_action_condition"):
        serve_policy_proposal._apply_policy_config_overrides(
            MissingFieldConfig(), _args(action_condition="zero_action")
        )

    with pytest.raises(ValueError, match="scheme4_condition_negative_modes"):
        serve_policy_proposal._apply_policy_config_overrides(
            MissingFieldConfig(), _args(condition_negative_modes="language")
        )

    with pytest.raises(ValueError, match="scheme4_inference_lookahead_mode"):
        serve_policy_proposal._apply_policy_config_overrides(
            MissingFieldConfig(), _args(lookahead_mode="soft")
        )
