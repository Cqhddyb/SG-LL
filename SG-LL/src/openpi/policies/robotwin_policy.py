import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


def make_robotwin_example() -> dict:
    """Creates a random input example for the RoboTwin Aloha-AgileX policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "adjust the bottle",
    }


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Inputs for RoboTwin 2.0 Aloha-AgileX pi0.5 policies.

    Expected per-sample inputs before model transforms:
    - images.cam_high, images.cam_left_wrist, images.cam_right_wrist: uint8 or float image, [3,H,W] or [H,W,3]
    - state: [14]
    - actions: optional [H_action,14]
    - teacher_images/future_teacher_images: optional dicts with the same three camera keys
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )
    CAMERA_MAP: ClassVar[dict[str, str]] = {
        "base_0_rgb": "cam_high",
        "left_wrist_0_rgb": "cam_left_wrist",
        "right_wrist_0_rgb": "cam_right_wrist",
    }

    def __call__(self, data: dict) -> dict:
        in_images = _require_camera_dict(data, "images")
        state = _as_state(data["state"])

        # Model image keys are the standard pi0.5 keys; RoboTwin source keys are preserved only at the policy boundary.
        images = {dest: _as_hwc_uint8(in_images[src], name=f"images.{src}") for dest, src in self.CAMERA_MAP.items()}
        image_masks = dict.fromkeys(self.CAMERA_MAP, np.True_)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
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
class RoboTwinOutputs(transforms.DataTransformFn):
    """Outputs for RoboTwin 2.0 Aloha-AgileX policies."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"RoboTwin output actions must be rank 2 [H_action,A_dim], got {actions.shape}")
        if actions.shape[-1] < 14:
            raise ValueError(f"RoboTwin output actions need at least 14 dims, got {actions.shape[-1]}")
        return {"actions": actions[:, :14]}


def _convert_teacher_images(image_dict: dict, *, name: str) -> dict[str, np.ndarray]:
    in_images = _require_camera_dict({"images": image_dict}, "images")
    return {dest: _as_hwc_uint8(in_images[src], name=f"{name}.{src}") for dest, src in RoboTwinInputs.CAMERA_MAP.items()}


def _require_camera_dict(data: dict, key: str) -> dict:
    if key not in data:
        raise KeyError(f"RoboTwin input is missing required key '{key}'")
    images = data[key]
    if not isinstance(images, dict):
        raise TypeError(f"RoboTwin '{key}' must be a dict, got {type(images).__name__}")
    missing = set(RoboTwinInputs.EXPECTED_CAMERAS) - set(images)
    extra = set(images) - set(RoboTwinInputs.EXPECTED_CAMERAS)
    if missing or extra:
        raise ValueError(
            f"RoboTwin '{key}' must contain exactly {RoboTwinInputs.EXPECTED_CAMERAS}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return images


def _as_state(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (14,):
        raise ValueError(f"RoboTwin state must have shape [14], got {state.shape}")
    return state


def _as_actions(actions) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"RoboTwin actions must be rank 2 [H_action,14], got {actions.shape}")
    if actions.shape[-1] != 14:
        raise ValueError(f"RoboTwin actions last dim must be 14, got {actions.shape[-1]}")
    return actions


def _as_hwc_uint8(image, *, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"{name} must be rank 3 [3,H,W] or [H,W,3], got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] != 3:
        raise ValueError(f"{name} must be channel-first or channel-last RGB, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        min_value = float(np.nanmin(image))
        max_value = float(np.nanmax(image))
        if min_value < -1.01 or max_value > 255.0:
            raise ValueError(f"{name} float image values are outside supported ranges: [{min_value}, {max_value}]")
        if min_value < 0.0:
            image = (image + 1.0) * 127.5
        elif max_value <= 1.01:
            image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] != 3:
        raise ValueError(f"{name} converted image must be HWC RGB, got {image.shape}")
    return np.ascontiguousarray(image)
