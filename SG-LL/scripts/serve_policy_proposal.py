"""Serve a checkpoint trained with proposal variants.

This wraps the normal policy server and monkey-patches the geometry model class
before checkpoint loading, so no baseline openpi files need to be edited.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
import sys
from typing import Literal

import torch
import tyro

from openpi.models_pytorch import pi0_align_pytorch
from openpi.models_pytorch.pi0_proposal_variants_pytorch import get_proposal_model_class


def _load_base_serve_module():
    script_path = Path(__file__).resolve().parent / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("_openpi_serve_policy", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base policy server from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_serve_policy = _load_base_serve_module()


@dataclasses.dataclass
class Args(_serve_policy.Args):
    proposal_variant: Literal[
        "scheme1",
        "scheme1_rq",
        "scheme2",
        "scheme2_g",
        "scheme2_unified",
        "scheme2_unified_aux",
        "direct_vggt_injection",
        "scheme3_d3",
        "scheme3_delta",
        "scheme3_delta_intent",
        "scheme4_lac_double_align",
        "scheme4_grounded_lac_double_align",
        "scheme4_frame_token_lac_double_align",
    ] = "scheme1"

    # Optional proposal-policy config overrides needed for timing ablations.
    # They are no-ops unless explicitly passed.
    policy_data_future_teacher_obs_offset: int | None = None
    policy_scheme3_horizon_id: int | None = None
    policy_scheme4_delta_enabled: Literal["default", "true", "false"] = "default"
    policy_scheme4_training_action_condition: Literal["default", "normal", "zero_action"] = "default"
    # Needed when a checkpoint's training-time action condition differs from
    # the registered base config used to reconstruct the policy at serving.
    policy_scheme4_condition_negative_modes: Literal["default", "language", "language,action"] = "default"
    policy_scheme4_inference_condition_ablation: Literal[
        "default",
        "none",
        "zero_language",
        "zero_action",
        "zero_language_action",
        "wrong_language",
        "wrong_action",
        "shuffled_language",
        "shuffled_action",
    ] = "default"
    policy_scheme4_inference_delta_control: Literal[
        "default",
        "normal",
        "zero_delta",
        "mean_delta",
        "random_delta",
        "norm_matched_random_delta",
        "cross_episode_delta",
        "batch_shuffled_delta",
    ] = "default"
    policy_scheme4_inference_delta_mean_path: str | None = None
    policy_scheme4_inference_control_seed: int | None = None
    policy_scheme4_inference_fusion_scale: float | None = None
    policy_scheme4_inference_diagnostics_enabled: Literal["default", "true", "false"] = "default"
    policy_scheme4_lookahead_mode: Literal["default", "none", "fixed", "soft", "hard"] = "default"
    policy_scheme4_tokenizer_checkpoint: str | None = None
    policy_torch_seed: int | None = None


def _apply_policy_config_overrides(train_config, args: Args):
    replace_kwargs = {}
    if args.policy_data_future_teacher_obs_offset is not None:
        if not hasattr(train_config.data, "future_teacher_obs_offset"):
            raise ValueError(
                "--policy-data-future-teacher-obs-offset was set, but this policy data config has no "
                "future_teacher_obs_offset field."
            )
        replace_kwargs["data"] = dataclasses.replace(
            train_config.data,
            future_teacher_obs_offset=int(args.policy_data_future_teacher_obs_offset),
        )
    if args.policy_scheme3_horizon_id is not None:
        replace_kwargs["scheme3_horizon_id"] = int(args.policy_scheme3_horizon_id)
    if args.policy_scheme4_delta_enabled != "default":
        if not hasattr(train_config, "scheme4_delta_enabled"):
            raise ValueError(
                "--policy-scheme4-delta-enabled was set, but this policy config has no scheme4_delta_enabled field."
            )
        scheme4_delta_enabled = args.policy_scheme4_delta_enabled == "true"
        replace_kwargs["scheme4_delta_enabled"] = scheme4_delta_enabled
        if not scheme4_delta_enabled:
            # These losses are train-only, but the Scheme4 constructor rejects a
            # delta-disabled config unless they are explicitly disabled too.
            replace_kwargs["scheme4_delta_token_loss_coeff"] = 0.0
            replace_kwargs["scheme4_future_recon_loss_coeff"] = 0.0
    if args.policy_scheme4_training_action_condition != "default":
        if not hasattr(train_config, "scheme4_training_action_condition"):
            raise ValueError(
                "--policy-scheme4-training-action-condition was set, but this policy config has no "
                "scheme4_training_action_condition field."
            )
        replace_kwargs["scheme4_training_action_condition"] = args.policy_scheme4_training_action_condition
    if args.policy_scheme4_condition_negative_modes != "default":
        if not hasattr(train_config, "scheme4_condition_negative_modes"):
            raise ValueError(
                "--policy-scheme4-condition-negative-modes was set, but this policy config has no "
                "scheme4_condition_negative_modes field."
            )
        replace_kwargs["scheme4_condition_negative_modes"] = args.policy_scheme4_condition_negative_modes
    if args.policy_scheme4_inference_condition_ablation != "default":
        if not hasattr(train_config, "scheme4_inference_condition_ablation"):
            raise ValueError(
                "--policy-scheme4-inference-condition-ablation was set, but this policy config has no "
                "scheme4_inference_condition_ablation field."
            )
        replace_kwargs["scheme4_inference_condition_ablation"] = args.policy_scheme4_inference_condition_ablation
    if args.policy_scheme4_inference_delta_control != "default":
        if not hasattr(train_config, "scheme4_inference_delta_control"):
            raise ValueError(
                "--policy-scheme4-inference-delta-control was set, but this policy config has no "
                "scheme4_inference_delta_control field."
            )
        replace_kwargs["scheme4_inference_delta_control"] = args.policy_scheme4_inference_delta_control
    if args.policy_scheme4_inference_delta_mean_path is not None:
        if not hasattr(train_config, "scheme4_inference_delta_mean_path"):
            raise ValueError(
                "--policy-scheme4-inference-delta-mean-path was set, but this policy config has no "
                "scheme4_inference_delta_mean_path field."
            )
        replace_kwargs["scheme4_inference_delta_mean_path"] = args.policy_scheme4_inference_delta_mean_path
    if args.policy_scheme4_inference_control_seed is not None:
        if not hasattr(train_config, "scheme4_inference_control_seed"):
            raise ValueError(
                "--policy-scheme4-inference-control-seed was set, but this policy config has no "
                "scheme4_inference_control_seed field."
            )
        replace_kwargs["scheme4_inference_control_seed"] = int(args.policy_scheme4_inference_control_seed)
    if args.policy_scheme4_inference_fusion_scale is not None:
        if not hasattr(train_config, "scheme4_inference_fusion_scale"):
            raise ValueError(
                "--policy-scheme4-inference-fusion-scale was set, but this policy config has no "
                "scheme4_inference_fusion_scale field."
            )
        fusion_scale = float(args.policy_scheme4_inference_fusion_scale)
        if not math.isfinite(fusion_scale) or fusion_scale < 0.0:
            raise ValueError("--policy-scheme4-inference-fusion-scale must be finite and non-negative")
        replace_kwargs["scheme4_inference_fusion_scale"] = fusion_scale
    if args.policy_scheme4_inference_diagnostics_enabled != "default":
        if not hasattr(train_config, "scheme4_inference_diagnostics_enabled"):
            raise ValueError(
                "--policy-scheme4-inference-diagnostics-enabled was set, but this policy config has no "
                "scheme4_inference_diagnostics_enabled field."
            )
        replace_kwargs["scheme4_inference_diagnostics_enabled"] = (
            args.policy_scheme4_inference_diagnostics_enabled == "true"
        )
    if args.policy_scheme4_lookahead_mode != "default":
        if not hasattr(train_config, "scheme4_inference_lookahead_mode"):
            raise ValueError(
                "--policy-scheme4-lookahead-mode was set, but this policy config has no "
                "scheme4_inference_lookahead_mode field."
            )
        replace_kwargs["scheme4_inference_lookahead_mode"] = args.policy_scheme4_lookahead_mode
        if args.policy_scheme4_lookahead_mode in {"soft", "hard"}:
            replace_kwargs["scheme4_adaptive_lookahead_enabled"] = True
    if args.policy_scheme4_tokenizer_checkpoint is not None:
        if not hasattr(train_config, "scheme4_tokenizer_checkpoint"):
            raise ValueError(
                "--policy-scheme4-tokenizer-checkpoint was set, but this policy config has no "
                "scheme4_tokenizer_checkpoint field."
            )
        replace_kwargs["scheme4_tokenizer_checkpoint"] = args.policy_scheme4_tokenizer_checkpoint
    if not replace_kwargs:
        return train_config
    return dataclasses.replace(train_config, **replace_kwargs)


def _apply_checkpoint_architecture(train_config, checkpoint_config: dict):
    """Restore optional modules needed to load the checkpoint state dict."""
    key = "scheme4_adaptive_lookahead_enabled"
    if key not in checkpoint_config:
        return train_config
    if not hasattr(train_config, key):
        raise ValueError(f"Checkpoint metadata has {key}, but the serving config does not.")
    return dataclasses.replace(train_config, **{key: bool(checkpoint_config[key])})


def _create_policy_with_overrides(args: Args):
    match args.policy:
        case _serve_policy.Checkpoint():
            train_config = _serve_policy._config.get_config(args.policy.config)  # noqa: SLF001
            train_config = _apply_policy_config_overrides(train_config, args)
            if args.policy_scheme4_lookahead_mode != "default":
                metadata_path = Path(args.policy.dir) / "metadata.pt"
                if args.policy_scheme4_lookahead_mode in {"soft", "hard"} and not metadata_path.is_file():
                    raise FileNotFoundError(
                        "hard/soft adaptive lookahead requires checkpoint metadata proving the router was trained: "
                        f"{metadata_path}"
                    )
                checkpoint_config = {}
                if metadata_path.is_file():
                    checkpoint_metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
                    checkpoint_config = (
                        checkpoint_metadata.get("config", {}) if isinstance(checkpoint_metadata, dict) else {}
                    )
                    train_config = _apply_checkpoint_architecture(train_config, checkpoint_config)
                if args.policy_scheme4_lookahead_mode in {"soft", "hard"} and not checkpoint_config.get(
                    "scheme4_adaptive_lookahead_enabled", False
                ):
                    raise ValueError(
                        "hard/soft adaptive lookahead requires a checkpoint trained with "
                        "scheme4_adaptive_lookahead_enabled=True; refusing random router weights"
                    )
            if (
                args.policy_data_future_teacher_obs_offset is not None
                or args.policy_scheme3_horizon_id is not None
                or args.policy_scheme4_delta_enabled != "default"
                or args.policy_scheme4_training_action_condition != "default"
                or args.policy_scheme4_condition_negative_modes != "default"
                or args.policy_scheme4_inference_condition_ablation != "default"
                or args.policy_scheme4_inference_delta_control != "default"
                or args.policy_scheme4_inference_delta_mean_path is not None
                or args.policy_scheme4_inference_control_seed is not None
                or args.policy_scheme4_inference_fusion_scale is not None
                or args.policy_scheme4_inference_diagnostics_enabled != "default"
                or args.policy_scheme4_lookahead_mode != "default"
                or args.policy_scheme4_tokenizer_checkpoint is not None
            ):
                _serve_policy.logging.info(
                    "Applied proposal policy overrides: future_teacher_obs_offset=%s, "
                    "scheme3_horizon_id=%s, scheme4_delta_enabled=%s, scheme4_training_action_condition=%s, "
                    "scheme4_condition_negative_modes=%s, "
                    "scheme4_inference_condition_ablation=%s, scheme4_inference_delta_control=%s, "
                    "scheme4_inference_delta_mean_path=%s, scheme4_inference_control_seed=%s, "
                    "scheme4_inference_fusion_scale=%s, scheme4_inference_diagnostics_enabled=%s, "
                    "scheme4_lookahead_mode=%s, scheme4_tokenizer_checkpoint=%s",
                    args.policy_data_future_teacher_obs_offset,
                    args.policy_scheme3_horizon_id,
                    args.policy_scheme4_delta_enabled,
                    args.policy_scheme4_training_action_condition,
                    args.policy_scheme4_condition_negative_modes,
                    args.policy_scheme4_inference_condition_ablation,
                    args.policy_scheme4_inference_delta_control,
                    args.policy_scheme4_inference_delta_mean_path,
                    args.policy_scheme4_inference_control_seed,
                    args.policy_scheme4_inference_fusion_scale,
                    args.policy_scheme4_inference_diagnostics_enabled,
                    args.policy_scheme4_lookahead_mode,
                    args.policy_scheme4_tokenizer_checkpoint,
                )
            return _serve_policy._policy_config.create_trained_policy(  # noqa: SLF001
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case _serve_policy.Default():
            if (
                args.policy_data_future_teacher_obs_offset is not None
                or args.policy_scheme3_horizon_id is not None
                or args.policy_scheme4_delta_enabled != "default"
                or args.policy_scheme4_training_action_condition != "default"
                or args.policy_scheme4_condition_negative_modes != "default"
                or args.policy_scheme4_inference_condition_ablation != "default"
                or args.policy_scheme4_inference_delta_control != "default"
                or args.policy_scheme4_inference_delta_mean_path is not None
                or args.policy_scheme4_inference_control_seed is not None
                or args.policy_scheme4_inference_fusion_scale is not None
                or args.policy_scheme4_inference_diagnostics_enabled != "default"
                or args.policy_scheme4_lookahead_mode != "default"
                or args.policy_scheme4_tokenizer_checkpoint is not None
            ):
                raise ValueError("Proposal policy config overrides require policy:checkpoint.")
            return _serve_policy.create_default_policy(args.env, default_prompt=args.default_prompt)


def seed_policy_rng(seed: int | None) -> None:
    """Seed policy-side PyTorch sampling without changing the default path."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: Args) -> None:
    seed_policy_rng(args.policy_torch_seed)
    pi0_align_pytorch.PI0Pytorch = get_proposal_model_class(args.proposal_variant)
    _serve_policy.create_policy = _create_policy_with_overrides
    _serve_policy.main(args)


if __name__ == "__main__":
    _serve_policy.logging.basicConfig(level=_serve_policy.logging.INFO, force=True)
    main(tyro.cli(Args))
