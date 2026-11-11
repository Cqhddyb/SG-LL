import numpy as np
import pytest

from openpi.policies import calvin_policy


def _images(dtype=np.float32):
    top = np.linspace(0, 1, 3 * 10 * 12, dtype=np.float32).reshape(3, 10, 12)
    wrist = np.linspace(0, 1, 3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8)
    if dtype == np.uint8:
        top = (top * 255).astype(np.uint8)
        wrist = (wrist * 255).astype(np.uint8)
    return {"top": top.astype(dtype, copy=False), "wrist": wrist.astype(dtype, copy=False)}


def test_calvin_inputs_shapes_and_future_targets() -> None:
    output = calvin_policy.CalvinInputs()(
        {
            "images": _images(),
            "state": np.arange(15, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "teacher_images": _images(np.uint8),
            "future_teacher_images": _images(np.uint8),
            "future_teacher_fallback": np.asarray(False),
            "prompt": "open the drawer",
        }
    )

    assert output["state"].shape == (15,)
    assert output["actions"].shape == (10, 7)
    assert output["image"]["base_0_rgb"].shape == (10, 12, 3)
    assert output["image"]["left_wrist_0_rgb"].shape == (6, 8, 3)
    assert output["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert output["teacher_image"]["base_0_rgb"].shape == (10, 12, 3)
    assert output["future_teacher_image"]["left_wrist_0_rgb"].shape == (6, 8, 3)
    assert output["future_teacher_fallback"].shape == ()


def test_calvin_inputs_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match=r"shape \[15\]"):
        calvin_policy.CalvinInputs()({"images": _images(), "state": np.zeros((14,), dtype=np.float32)})
    with pytest.raises(ValueError, match=r"H_action,7"):
        calvin_policy.CalvinInputs()(
            {
                "images": _images(),
                "state": np.zeros((15,), dtype=np.float32),
                "actions": np.zeros((10, 8), dtype=np.float32),
            }
        )


def test_calvin_outputs_discretize_gripper_without_mutating_input() -> None:
    actions = np.zeros((3, 32), dtype=np.float32)
    actions[:, 6] = [-0.2, 0.0, 0.3]
    original = actions.copy()

    output = calvin_policy.CalvinOutputs()({"actions": actions})["actions"]

    assert output.shape == (3, 7)
    np.testing.assert_array_equal(output[:, 6], [-1.0, 1.0, 1.0])
    np.testing.assert_array_equal(actions, original)
