"""Strict JSONL row contract for Scheme4 LIBERO inference diagnostics."""

from __future__ import annotations

import math
from typing import Any

FLOW_METRICS = (
    "hidden_residual_l2_mean",
    "hidden_residual_l2_max",
    "hidden_base_l2_mean",
    "hidden_residual_to_base_ratio",
    "velocity_delta_l2_mean",
    "velocity_delta_l2_max",
    "velocity_base_l2_mean",
    "velocity_delta_to_base_ratio",
)
REQUEST_METRICS = ("pred_delta_token_l2_mean", "action_intent_l2", "fusion_scale")
LOOKAHEAD_METRICS = ("lookahead_probability", "lookahead_scale")


def validate_diagnostics_payload(payload: object) -> dict[str, float | list[float]]:
    if not isinstance(payload, dict):
        raise ValueError("diagnostics payload must be an object")
    expected = {*FLOW_METRICS, *REQUEST_METRICS}
    # Legacy fixed-horizon payloads remain valid. Adaptive payloads must include
    # both bounded routing fields; arbitrary or partial extensions are rejected.
    if set(payload).intersection(LOOKAHEAD_METRICS):
        expected.update(LOOKAHEAD_METRICS)
    if set(payload) != expected:
        raise ValueError(f"diagnostics keys differ: got={sorted(payload)}, expected={sorted(expected)}")
    validated: dict[str, float | list[float]] = {}
    flow_steps: int | None = None
    for name in FLOW_METRICS:
        values = payload[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"diagnostic {name} must be a non-empty flow-step list")
        floats = [float(value) for value in values]
        if not all(math.isfinite(value) and value >= 0.0 for value in floats):
            raise ValueError(f"diagnostic {name} must contain finite non-negative values")
        if flow_steps is None:
            flow_steps = len(floats)
        elif len(floats) != flow_steps:
            raise ValueError("all flow diagnostic vectors must have the same number of steps")
        validated[name] = floats
    for name in REQUEST_METRICS:
        value = float(payload[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"diagnostic {name} must be finite and non-negative")
        validated[name] = value
    if set(payload).intersection(LOOKAHEAD_METRICS):
        for name in LOOKAHEAD_METRICS:
            value = float(payload[name])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"diagnostic {name} must be finite and in [0, 1]")
            validated[name] = value
    return validated


def make_diagnostics_row(
    *,
    run_id: str,
    suite: str,
    task_id: int,
    episode_idx: int,
    seed: int,
    policy_request_index: int,
    env_step: int,
    replan_steps: int,
    diagnostics: object,
) -> dict[str, Any]:
    integer_values = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "seed": seed,
        "policy_request_index": policy_request_index,
        "env_step": env_step,
        "replan_steps": replan_steps,
    }
    if not run_id or not suite:
        raise ValueError("run_id and suite must be non-empty")
    if any(not isinstance(value, int) for value in integer_values.values()):
        raise ValueError("diagnostics row indices must be explicit integers")
    if min(task_id, episode_idx, policy_request_index, env_step) < 0 or replan_steps <= 0:
        raise ValueError("diagnostics row indices must be non-negative and replan_steps positive")
    return {
        "type": "scheme4_policy_request_diagnostics",
        "run_id": run_id,
        "suite": suite,
        **integer_values,
        "diagnostics": validate_diagnostics_payload(diagnostics),
    }
