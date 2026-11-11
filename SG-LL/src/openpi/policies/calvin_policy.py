"""CALVIN policy-boundary transforms for pi0.5 training and inference."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


def make_calvin_example() -> dict:
    """Create one inference-shaped CALVIN observation."""
    return {
        "state": np.zeros((15,), dtype=np.float32),
        "images": {
            "top": np.zeros((200, 200, 3), dtype=np.uint8),
            "wrist": np.zeros((84, 84, 3), dtype=np.uint8),
        },
        "prompt": "open the drawer",
    }


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    """Convert CALVIN observations into the fixed pi0.5 tensor contract.

    Input shapes are ``state[15]``, ``images.top[H,W,3]``,
    ``images.wrist[H,W,3]``, and optional ``actions[H_action,7]``. Future
    images are supervision-only and use the same two-camera structure.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top", "wrist")

    def __call__(self, data: dict) -> dict:
        images = _require_camera_dict(data, "images")
        top = _as_hwc_uint8(images["top"], name="images.top")
        wrist = _as_hwc_uint8(images["wrist"], name="images.wrist")

        inputs = {
            "state": _as_state(data["state"]),
            "image": {
                "base_0_rgb": top,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(top),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = _as_actions(data["actions"])
        if "teacher_images" in data:
            inputs["teacher_image"] = _convert_teacher_images(data["teacher_images"], name="teacher_images")
        if "future_teacher_images" in data:
            inputs["future_teacher_image"] = _convert_teacher_images(
                data["future_teacher_images"], name="future_teacher_images"
            )
        if "future_teacher_fallback" in data:
            inputs["future_teacher_fallback"] = np.asarray(data["future_teacher_fallback"], dtype=np.bool_)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """Return CALVIN's 6-D relative motion and discrete gripper command."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < 7:
            raise ValueError(f"CALVIN outputs require [H_action,A>=7], got {actions.shape}")
        outputs = np.array(actions[:, :7], dtype=np.float32, copy=True)
        outputs[:, 6] = np.where(outputs[:, 6] >= 0.0, 1.0, -1.0)
        return {"actions": outputs}


def _convert_teacher_images(image_dict: dict, *, name: str) -> dict[str, np.ndarray]:
    images = _require_camera_dict({"images": image_dict}, "images")
    top = _as_hwc_uint8(images["top"], name=f"{name}.top")
    wrist = _as_hwc_uint8(images["wrist"], name=f"{name}.wrist")
    return {
        "base_0_rgb": top,
        "left_wrist_0_rgb": wrist,
        "right_wrist_0_rgb": np.zeros_like(top),
    }


def _require_camera_dict(data: dict, key: str) -> dict:
    if key not in data:
        raise KeyError(f"CALVIN input is missing required key {key!r}")
    images = data[key]
    if not isinstance(images, dict):
        raise TypeError(f"CALVIN {key!r} must be a dict, got {type(images).__name__}")
    expected = set(CalvinInputs.EXPECTED_CAMERAS)
    if set(images) != expected:
        raise ValueError(
            f"CALVIN {key!r} must contain exactly {CalvinInputs.EXPECTED_CAMERAS}; "
            f"missing={sorted(expected - set(images))}, extra={sorted(set(images) - expected)}"
        )
    return images


def _as_state(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (15,):
        raise ValueError(f"CALVIN state must have shape [15], got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("CALVIN state contains non-finite values")
    return state


def _as_actions(actions) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"CALVIN actions must have shape [H_action,7], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("CALVIN actions contain non-finite values")
    return actions


def _as_hwc_uint8(image, *, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"{name} must be rank 3 RGB, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must be CHW or HWC RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains non-finite values")
        minimum = float(image.min())
        maximum = float(image.max())
        if minimum < -1.01 or maximum > 255.0:
            raise ValueError(f"{name} values are outside supported ranges: [{minimum}, {maximum}]")
        if minimum < 0.0:
            image = (image + 1.0) * 127.5
        elif maximum <= 1.01:
            image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
