import numpy as np
import pytest

from openpi.policies import simpler_env_policy


def test_simpler_inputs_shapes_masks_and_future_targets() -> None:
    image = np.linspace(0, 1, 3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
    output = simpler_env_policy.SimplerEnvInputs()(
        {
            "image": image,
            "state": np.arange(8, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "teacher_image": image,
            "future_teacher_image": image,
            "future_teacher_fallback": np.asarray(False),
            "prompt": "put the spoon on the towel",
        }
    )

    assert output["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert output["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert output["state"].shape == (8,)
    assert output["actions"].shape == (10, 7)
    assert output["teacher_image"]["base_0_rgb"].shape == (8, 10, 3)
    assert output["future_teacher_image"]["right_wrist_0_rgb"].shape == (8, 10, 3)


def test_simpler_inputs_reject_bad_state_and_action_shapes() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"shape \[8\]"):
        simpler_env_policy.SimplerEnvInputs()({"image": image, "state": np.zeros((7,), dtype=np.float32)})
    with pytest.raises(ValueError, match=r"H_action,7"):
        simpler_env_policy.SimplerEnvInputs()(
            {
                "image": image,
                "state": np.zeros((8,), dtype=np.float32),
                "actions": np.zeros((10, 8), dtype=np.float32),
            }
        )


def test_fedor_bridge_seven_dimensional_state_is_explicitly_padded() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    source = np.asarray([0.3, 0.1, 0.2, -0.1, 0.2, -0.3, 0.75], dtype=np.float32)

    output = simpler_env_policy.SimplerEnvInputs(state_layout="bridge_eef7_pad_zero")(
        {"image": image, "state": source}
    )

    np.testing.assert_array_equal(
        output["state"],
        np.asarray([0.3, 0.1, 0.2, -0.1, 0.2, -0.3, 0.0, 0.75], dtype=np.float32),
    )
    assert output["state"].shape == (8,)
    assert not np.shares_memory(output["state"], source)


def test_fedor_state_layout_rejects_implicit_eight_dimensional_input() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"bridge_eef7_pad_zero.*shape \[7\]"):
        simpler_env_policy.SimplerEnvInputs(state_layout="bridge_eef7_pad_zero")(
            {"image": image, "state": np.zeros((8,), dtype=np.float32)}
        )


def test_simpler_outputs_copy_first_seven_dims() -> None:
    actions = np.arange(2 * 32, dtype=np.float32).reshape(2, 32)
    output = simpler_env_policy.SimplerEnvOutputs()({"actions": actions})["actions"]
    assert output.shape == (2, 7)
    np.testing.assert_array_equal(output, actions[:, :7])
    assert not np.shares_memory(output, actions)


def test_bridge_actions_convert_euler_to_axis_angle_and_binarize_gripper() -> None:
    dataset_actions = np.asarray(
        [
            [0.01, -0.02, 0.03, 0.0, 0.0, np.pi / 2, 0.75],
            [-0.01, 0.02, -0.03, 0.0, 0.0, 0.0, 0.25],
        ],
        dtype=np.float32,
    )

    output = simpler_env_policy.SimplerEnvOutputs(
        action_layout="simpler_axis_angle_binary_gripper"
    )({"actions": dataset_actions})["actions"]

    np.testing.assert_allclose(output[:, :3], dataset_actions[:, :3], atol=0.0)
    np.testing.assert_allclose(output[0, 3:6], [0.0, 0.0, np.pi / 2], atol=1e-6)
    np.testing.assert_array_equal(output[1, 3:6], np.zeros((3,), dtype=np.float32))
    np.testing.assert_array_equal(output[:, 6], np.asarray([1.0, -1.0], dtype=np.float32))


def test_bridge_action_conversion_is_deterministic_and_rejects_nonfinite_values() -> None:
    action = np.asarray([0.0, 0.0, 0.0, 0.1, -0.2, 0.3, 0.5], dtype=np.float32)
    first = simpler_env_policy.bridge_dataset_actions_to_simpler(action)
    second = simpler_env_policy.bridge_dataset_actions_to_simpler(action.copy())
    np.testing.assert_array_equal(first, second)
    assert first.shape == (7,)
    assert first[6] == -1.0

    action[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        simpler_env_policy.bridge_dataset_actions_to_simpler(action)
