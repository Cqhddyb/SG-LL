"""Opt-in diagnostics for the RoboTwin policy boundary.

The normal RoboTwin policy path never constructs this writer. Diagnostic runs
use it to preserve the exact simulator-space actions, their corresponding
training-space representation, and a bounded number of camera observations.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CONTROL_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DELTA_JOINT_MASK = np.asarray(
    (True, True, True, True, True, True, False, True, True, True, True, True, True, False),
    dtype=np.bool_,
)
GRIPPER_INDICES = (6, 13)


def absolute_to_training_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert raw RoboTwin qpos targets to the representation normalized during training."""
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(CONTROL_NAMES):
        raise ValueError(f"actions must have shape [H,{len(CONTROL_NAMES)}], got {actions.shape}")
    if state.shape != (len(CONTROL_NAMES),):
        raise ValueError(f"state must have shape [{len(CONTROL_NAMES)}], got {state.shape}")

    state_rows = np.broadcast_to(state[np.newaxis, :], actions.shape)
    converted = actions.copy()
    converted[:, DELTA_JOINT_MASK] = actions[:, DELTA_JOINT_MASK] - state_rows[:, DELTA_JOINT_MASK]
    return converted


def summarize_array(values: np.ndarray) -> dict[str, Any]:
    """Return JSON-safe global and last-dimension statistics."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    finite_values = values[finite]
    summary: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(values.size - finite.sum()),
    }
    if finite_values.size:
        summary["global"] = {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
        }
    else:
        summary["global"] = None

    if values.ndim >= 1 and values.shape[-1] > 0:
        rows = values.reshape(-1, values.shape[-1])
        per_dim = []
        for column in rows.T:
            column_values = column[np.isfinite(column)]
            if column_values.size:
                per_dim.append(
                    {
                        "min": float(column_values.min()),
                        "max": float(column_values.max()),
                        "mean": float(column_values.mean()),
                        "std": float(column_values.std()),
                    }
                )
            else:
                per_dim.append(None)
        summary["per_dim"] = per_dim
    return summary


@dataclasses.dataclass
class RoboTwinDiagnosticWriter:
    """Write one JSONL row per policy request and a bounded RGB snapshot set."""

    output_dir: Path | str
    max_image_requests_per_episode: int = 1
    episode_index: int = dataclasses.field(default=-1, init=False)
    request_index: int = dataclasses.field(default=0, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.max_image_requests_per_episode < 0:
            raise ValueError("max_image_requests_per_episode must be non-negative")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def request_log_path(self) -> Path:
        return self.output_dir / "inference_requests.jsonl"

    def start_episode(self) -> None:
        self.episode_index += 1
        self.request_index = 0

    def record_request(
        self,
        *,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        actions: np.ndarray,
        instruction: str,
        action_chunk_size: int,
        executed_action_count: int,
        env_step_before: int,
        env_step_after: int,
    ) -> dict[str, Any]:
        if self.episode_index < 0:
            raise RuntimeError("start_episode() must be called before record_request()")
        if action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")
        if executed_action_count < 0 or executed_action_count > action_chunk_size:
            raise ValueError(
                "executed_action_count must be in [0, action_chunk_size], "
                f"got {executed_action_count} for chunk size {action_chunk_size}"
            )

        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        training_actions = absolute_to_training_actions(actions, state)
        checked_images = _validate_images(images)
        image_paths = self._save_images(checked_images)

        record = {
            "type": "robotwin_policy_request",
            "episode": self.episode_index,
            "request": self.request_index,
            "instruction": str(instruction),
            "control_order": list(CONTROL_NAMES),
            "delta_joint_mask": DELTA_JOINT_MASK.tolist(),
            "gripper_indices": list(GRIPPER_INDICES),
            "action_semantics": "returned actions are absolute qpos/gripper targets; training actions use joint deltas and absolute grippers",
            "action_chunk_size": int(action_chunk_size),
            "returned_action_count": int(actions.shape[0]),
            "executed_action_count": int(executed_action_count),
            "env_step_before": int(env_step_before),
            "env_step_after": int(env_step_after),
            "state": _json_safe_array(state),
            "returned_absolute_actions": _json_safe_array(actions),
            "returned_training_actions": _json_safe_array(training_actions),
            "first_absolute_action": _json_safe_array(actions[0]) if actions.shape[0] else None,
            "first_training_action": _json_safe_array(training_actions[0]) if actions.shape[0] else None,
            "state_summary": summarize_array(state),
            "absolute_action_summary": summarize_array(actions),
            "training_action_summary": summarize_array(training_actions),
            "camera_order": list(CAMERA_NAMES),
            "camera_summaries": {name: summarize_array(image) for name, image in checked_images.items()},
            "saved_images": image_paths,
        }
        with self.request_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        self.request_index += 1
        return record

    def _save_images(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        if self.request_index >= self.max_image_requests_per_episode:
            return {}
        paths = {}
        for name in CAMERA_NAMES:
            filename = f"episode{self.episode_index:03d}_request{self.request_index:03d}_{name}.png"
            path = self.output_dir / filename
            Image.fromarray(images[name]).save(path)
            paths[name] = str(path)
        return paths


def analyze_records(records: list[dict[str, Any]], norm_stats: dict[str, Any]) -> dict[str, Any]:
    """Aggregate request diagnostics and compare them with training normalization quantiles."""
    if not records:
        raise ValueError("at least one diagnostic record is required")
    absolute_chunks = [np.asarray(record["returned_absolute_actions"], dtype=np.float32) for record in records]
    training_chunks = [np.asarray(record["returned_training_actions"], dtype=np.float32) for record in records]
    states = np.asarray([record["state"] for record in records], dtype=np.float32)
    for actions in (*absolute_chunks, *training_chunks):
        if actions.ndim != 2 or actions.shape[1] != len(CONTROL_NAMES):
            raise ValueError(f"diagnostic action chunks must have shape [H,{len(CONTROL_NAMES)}], got {actions.shape}")
    absolute_actions = np.concatenate(absolute_chunks, axis=0)
    training_actions = np.concatenate(training_chunks, axis=0)

    normalized = norm_stats.get("norm_stats", norm_stats)
    action_reference = normalized["actions"]
    state_reference = normalized["state"]
    first_training_actions = np.asarray([chunk[0] for chunk in training_chunks if len(chunk)], dtype=np.float32)
    arm_first_actions = first_training_actions[:, DELTA_JOINT_MASK]

    return {
        "episodes": len({int(record["episode"]) for record in records}),
        "requests": len(records),
        "returned_actions": int(sum(len(chunk) for chunk in absolute_chunks)),
        "executed_actions": int(sum(int(record["executed_action_count"]) for record in records)),
        "env_steps": int(sum(int(record["env_step_after"]) - int(record["env_step_before"]) for record in records)),
        "chunk_sizes": sorted({int(record["action_chunk_size"]) for record in records}),
        "returned_action_counts": sorted({int(record["returned_action_count"]) for record in records}),
        "control_order": list(CONTROL_NAMES),
        "absolute_action_summary": summarize_array(absolute_actions),
        "training_action_summary": summarize_array(training_actions),
        "state_summary": summarize_array(states),
        "training_action_reference_check": _quantile_reference_check(training_actions, action_reference),
        "state_reference_check": _quantile_reference_check(states, state_reference),
        "first_action_arm_delta_abs_max": float(np.abs(arm_first_actions).max()) if arm_first_actions.size else None,
        "first_action_arm_delta_abs_mean": float(np.abs(arm_first_actions).mean()) if arm_first_actions.size else None,
        "gripper_outside_unit_interval": {
            str(index): int(((absolute_actions[:, index] < 0.0) | (absolute_actions[:, index] > 1.0)).sum())
            for index in GRIPPER_INDICES
        },
        "saved_images": [path for record in records for path in record.get("saved_images", {}).values()],
        "camera_summaries_first_request": records[0]["camera_summaries"],
    }


def _quantile_reference_check(values: np.ndarray, reference: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(CONTROL_NAMES):
        raise ValueError(f"reference values must have shape [N,{len(CONTROL_NAMES)}], got {values.shape}")
    q01 = np.asarray(reference["q01"], dtype=np.float32)
    q99 = np.asarray(reference["q99"], dtype=np.float32)
    if q01.shape != (len(CONTROL_NAMES),) or q99.shape != (len(CONTROL_NAMES),):
        raise ValueError(f"reference q01/q99 must have shape [{len(CONTROL_NAMES)}]")
    q01_rows = np.broadcast_to(q01[np.newaxis, :], values.shape)
    q99_rows = np.broadcast_to(q99[np.newaxis, :], values.shape)
    finite = np.isfinite(values)
    below = finite & (values < q01_rows)
    above = finite & (values > q99_rows)
    denominator = np.maximum(finite.sum(axis=0), 1)
    outside = below | above
    return {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "below_count_per_dim": below.sum(axis=0).astype(int).tolist(),
        "above_count_per_dim": above.sum(axis=0).astype(int).tolist(),
        "outside_fraction_per_dim": (outside.sum(axis=0) / denominator).tolist(),
        "outside_fraction_total": float(outside.sum() / max(int(finite.sum()), 1)),
        "nonfinite_count": int(values.size - finite.sum()),
    }


def _validate_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = set(CAMERA_NAMES) - set(images)
    extra = set(images) - set(CAMERA_NAMES)
    if missing or extra:
        raise ValueError(
            f"images must contain exactly {CAMERA_NAMES}; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    checked = {}
    for name in CAMERA_NAMES:
        image = np.asarray(images[name])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [H,W,3], got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {image.dtype}")
        checked[name] = np.ascontiguousarray(image)
    return checked


def _json_safe_array(values: np.ndarray) -> list[Any]:
    values = np.asarray(values)
    if np.isfinite(values).all():
        return values.tolist()
    safe = values.astype(object)
    safe[~np.isfinite(values)] = None
    return safe.tolist()
