"""Detached inference diagnostics for Scheme4 residual fusion."""

from __future__ import annotations

import math

import torch

STEP_METRICS = (
    "hidden_residual_l2_mean",
    "hidden_residual_l2_max",
    "hidden_base_l2_mean",
    "hidden_residual_to_base_ratio",
    "velocity_delta_l2_mean",
    "velocity_delta_l2_max",
    "velocity_base_l2_mean",
    "velocity_delta_to_base_ratio",
)


def flow_step_diagnostics(
    suffix_hidden: torch.Tensor,
    hidden_residual: torch.Tensor,
    full_velocity: torch.Tensor,
    base_velocity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return one detached `[B]` metric vector per diagnostic."""

    if suffix_hidden.ndim != 3:
        raise ValueError(f"suffix_hidden must be [B,H,D], got {tuple(suffix_hidden.shape)}")
    if hidden_residual.shape != suffix_hidden.shape:
        raise ValueError(
            "hidden_residual must exactly match suffix_hidden; "
            f"got {tuple(hidden_residual.shape)} and {tuple(suffix_hidden.shape)}"
        )
    if full_velocity.ndim != 3:
        raise ValueError(f"full_velocity must be [B,H,A], got {tuple(full_velocity.shape)}")
    if base_velocity.shape != full_velocity.shape:
        raise ValueError(
            "base_velocity must exactly match full_velocity; "
            f"got {tuple(base_velocity.shape)} and {tuple(full_velocity.shape)}"
        )
    if full_velocity.shape[:2] != suffix_hidden.shape[:2]:
        raise ValueError(
            "velocity and hidden tensors must share exact [B,H]; "
            f"got {tuple(full_velocity.shape[:2])} and {tuple(suffix_hidden.shape[:2])}"
        )

    suffix_l2 = suffix_hidden.float().norm(dim=-1)
    residual_l2 = hidden_residual.float().norm(dim=-1)
    velocity_l2 = base_velocity.float().norm(dim=-1)
    velocity_delta_l2 = (full_velocity.float() - base_velocity.float()).norm(dim=-1)
    epsilon = torch.finfo(torch.float32).eps
    metrics = {
        "hidden_residual_l2_mean": residual_l2.mean(dim=1),
        "hidden_residual_l2_max": residual_l2.max(dim=1).values,
        "hidden_base_l2_mean": suffix_l2.mean(dim=1),
        "hidden_residual_to_base_ratio": residual_l2.mean(dim=1) / suffix_l2.mean(dim=1).clamp_min(epsilon),
        "velocity_delta_l2_mean": velocity_delta_l2.mean(dim=1),
        "velocity_delta_l2_max": velocity_delta_l2.max(dim=1).values,
        "velocity_base_l2_mean": velocity_l2.mean(dim=1),
        "velocity_delta_to_base_ratio": velocity_delta_l2.mean(dim=1) / velocity_l2.mean(dim=1).clamp_min(epsilon),
    }
    for name, value in metrics.items():
        if value.shape != (suffix_hidden.shape[0],):
            raise RuntimeError(f"{name} must be [B], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")
        metrics[name] = value.detach()
    return metrics


def finalize_request_diagnostics(
    step_metrics: list[dict[str, torch.Tensor]],
    *,
    pred_delta_tokens: torch.Tensor,
    action_intent: torch.Tensor,
    fusion_scale: float,
) -> dict[str, torch.Tensor]:
    """Stack step vectors into `[B,S]` and add request-level `[B]` values."""

    if not step_metrics:
        raise ValueError("step_metrics must contain at least one flow step")
    if pred_delta_tokens.ndim != 3:
        raise ValueError(f"pred_delta_tokens must be [B,M,D], got {tuple(pred_delta_tokens.shape)}")
    if action_intent.ndim != 2:
        raise ValueError(f"action_intent must be [B,D], got {tuple(action_intent.shape)}")
    batch_size = pred_delta_tokens.shape[0]
    if action_intent.shape[0] != batch_size:
        raise ValueError("pred_delta_tokens and action_intent must have the same batch size")
    if not math.isfinite(fusion_scale) or fusion_scale < 0.0:
        raise ValueError(f"fusion_scale must be finite and non-negative, got {fusion_scale}")

    output: dict[str, torch.Tensor] = {}
    for name in STEP_METRICS:
        values = []
        for step_index, metrics in enumerate(step_metrics):
            if set(metrics) != set(STEP_METRICS):
                raise ValueError(f"flow step {step_index} has unexpected metric keys: {sorted(metrics)}")
            value = metrics[name]
            if value.shape != (batch_size,):
                raise ValueError(f"flow step {step_index} metric {name} must be [B], got {tuple(value.shape)}")
            values.append(value.float())
        stacked = torch.stack(values, dim=1).detach()
        if not torch.isfinite(stacked).all():
            raise RuntimeError(f"stacked metric {name} contains non-finite values")
        output[name] = stacked

    output["pred_delta_token_l2_mean"] = pred_delta_tokens.float().norm(dim=-1).mean(dim=1).detach()
    output["action_intent_l2"] = action_intent.float().norm(dim=-1).detach()
    output["fusion_scale"] = torch.full(
        (batch_size,),
        float(fusion_scale),
        dtype=torch.float32,
        device=pred_delta_tokens.device,
    )
    return output
