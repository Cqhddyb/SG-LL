import numpy as np
import pytest

from openpi.policies import robotwin_policy


def _sample_images(dtype=np.float32):
    image = np.linspace(0.0, 1.0, 3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
    if dtype == np.uint8:
        image = (image * 255).astype(np.uint8)
    return {
        "cam_high": image.astype(dtype, copy=False),
        "cam_left_wrist": image.astype(dtype, copy=False),
        "cam_right_wrist": image.astype(dtype, copy=False),
    }


def test_robotwin_inputs_training_shapes_with_teacher_images():
    transform = robotwin_policy.RoboTwinInputs()
    item = {
        "images": _sample_images(),
        "state": np.arange(14, dtype=np.float32),
        "actions": np.zeros((50, 14), dtype=np.float32),
        "teacher_images": _sample_images(np.uint8),
        "future_teacher_images": _sample_images(np.uint8),
        "future_teacher_fallback": np.asarray(0, dtype=np.bool_),
        "prompt": "adjust the bottle",
    }

    out = transform(item)

    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert out["image"]["base_0_rgb"].dtype == np.uint8
    assert out["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }
    assert out["state"].shape == (14,)
    assert out["actions"].shape == (50, 14)
    assert out["teacher_image"]["base_0_rgb"].shape == (8, 10, 3)
    assert out["future_teacher_image"]["right_wrist_0_rgb"].shape == (8, 10, 3)
    assert out["future_teacher_fallback"].shape == ()


def test_robotwin_inputs_reject_bad_action_shape():
    transform = robotwin_policy.RoboTwinInputs()
    with pytest.raises(ValueError, match="actions"):
        transform(
            {
                "images": _sample_images(),
                "state": np.zeros((14,), dtype=np.float32),
                "actions": np.zeros((50, 13), dtype=np.float32),
                "prompt": "adjust the bottle",
            }
        )


def test_robotwin_outputs_returns_first_14_dims():
    transform = robotwin_policy.RoboTwinOutputs()
    actions = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)

    out = transform({"actions": actions})

    assert out["actions"].shape == (50, 14)
    np.testing.assert_array_equal(out["actions"], actions[:, :14])
