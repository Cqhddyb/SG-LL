"""SimplerEnv single-camera policy transforms for pi0.5."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from openpi import transforms
from openpi.policies.calvin_policy import _as_hwc_uint8

SimplerStateLayout = Literal["bridge_eef8", "bridge_eef7_pad_zero"]
SimplerActionLayout = Literal["bridge_euler_open", "simpler_axis_angle_binary_gripper"]


@dataclasses.dataclass(frozen=True)
class SimplerEnvInputs(transforms.DataTransformFn):
    """Map one RGB stream, 8-D state, and optional 7-D actions to pi0.5."""

    state_layout: SimplerStateLayout = "bridge_eef8"

    def __call__(self, data: dict) -> dict:
        image = _as_hwc_uint8(data["image"], name="image")
        state = _convert_state(data["state"], layout=self.state_layout)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"SimplerEnv actions must have shape [H_action,7], got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("SimplerEnv actions contain non-finite values")
            inputs["actions"] = actions
        if "teacher_image" in data:
            inputs["teacher_image"] = _teacher_images(data["teacher_image"], name="teacher_image")
        if "future_teacher_image" in data:
            inputs["future_teacher_image"] = _teacher_images(
                data["future_teacher_image"], name="future_teacher_image"
            )
        if "future_teacher_fallback" in data:
            inputs["future_teacher_fallback"] = np.asarray(data["future_teacher_fallback"], dtype=np.bool_)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SimplerEnvOutputs(transforms.DataTransformFn):
    """Expose dataset actions or explicitly convert them to SimplerEnv actions."""

    action_layout: SimplerActionLayout = "bridge_euler_open"

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < 7:
            raise ValueError(f"SimplerEnv outputs require [H_action,A>=7], got {actions.shape}")
        dataset_actions = np.array(actions[:, :7], dtype=np.float32, copy=True)
        if self.action_layout == "bridge_euler_open":
            return {"actions": dataset_actions}
        if self.action_layout == "simpler_axis_angle_binary_gripper":
            return {"actions": bridge_dataset_actions_to_simpler(dataset_actions)}
        raise ValueError(f"Unsupported SimplerEnv action layout: {self.action_layout!r}")


def bridge_dataset_actions_to_simpler(actions: np.ndarray) -> np.ndarray:
    """Convert Bridge ``[dxyz, dEulerXYZ, open]`` actions to SimplerEnv.

    Input and output both have shape ``[..., 7]``. The output contains
    ``[dxyz, rotation_vector, gripper]``; gripper is exactly ``-1`` (closed)
    or ``+1`` (open), following SimplerEnv's official WidowX policy adapter.
    """
    source = np.asarray(actions, dtype=np.float32)
    if source.ndim not in (1, 2) or source.shape[-1] != 7:
        raise ValueError(f"Bridge actions must have shape [7] or [H,7], got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("Bridge actions contain non-finite values")

    flat = source.reshape(-1, 7)
    converted = np.empty(flat.shape, dtype=np.float32)
    converted[:, :3] = flat[:, :3]
    converted[:, 3:6] = _euler_xyz_to_rotation_vector(flat[:, 3:6])
    converted[:, 6] = np.where(flat[:, 6] > 0.5, 1.0, -1.0).astype(np.float32)
    return converted.reshape(source.shape)


def _convert_state(state: np.ndarray, *, layout: SimplerStateLayout) -> np.ndarray:
    source = np.asarray(state, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("SimplerEnv state contains non-finite values")
    if layout == "bridge_eef8":
        if source.shape != (8,):
            raise ValueError(f"SimplerEnv bridge_eef8 state must have shape [8], got {source.shape}")
        return np.array(source, dtype=np.float32, copy=True)
    if layout == "bridge_eef7_pad_zero":
        if source.shape != (7,):
            raise ValueError(f"SimplerEnv bridge_eef7_pad_zero state must have shape [7], got {source.shape}")
        padded = np.empty((8,), dtype=np.float32)
        padded[:6] = source[:6]
        padded[6] = 0.0
        padded[7] = source[6]
        return padded
    raise ValueError(f"Unsupported SimplerEnv state layout: {layout!r}")


def _euler_xyz_to_rotation_vector(euler_xyz: np.ndarray) -> np.ndarray:
    """Match ``transforms3d.euler.euler2axangle`` for static XYZ Euler angles."""
    euler = np.asarray(euler_xyz, dtype=np.float64)
    if euler.ndim != 2 or euler.shape[1] != 3:
        raise ValueError(f"Euler rotations must have shape [N,3], got {euler.shape}")

    half_roll = euler[:, 0] * 0.5
    half_pitch = euler[:, 1] * 0.5
    half_yaw = euler[:, 2] * 0.5
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)

    quaternion = np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=1,
    )
    quaternion_norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.repeat(quaternion_norm, 4, axis=1)
    negative_scalar = quaternion[:, 0] < 0.0
    quaternion[negative_scalar] = np.negative(quaternion[negative_scalar])

    vector = quaternion[:, 1:]
    vector_norm = np.linalg.norm(vector, axis=1)
    rotation_vector = np.zeros_like(vector)
    nonzero = vector_norm > 1e-12
    angle = 2.0 * np.arctan2(vector_norm[nonzero], quaternion[nonzero, 0])
    scale = np.repeat((angle / vector_norm[nonzero])[:, None], 3, axis=1)
    rotation_vector[nonzero] = vector[nonzero] * scale
    return rotation_vector.astype(np.float32)


def _teacher_images(image, *, name: str) -> dict[str, np.ndarray]:
    image = _as_hwc_uint8(image, name=name)
    return {
        "base_0_rgb": image,
        "left_wrist_0_rgb": np.zeros_like(image),
        "right_wrist_0_rgb": np.zeros_like(image),
    }
