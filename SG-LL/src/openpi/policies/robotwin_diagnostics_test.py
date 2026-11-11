import json

import numpy as np
from PIL import Image

from openpi.policies import robotwin_diagnostics


def _images(value: int = 20) -> dict[str, np.ndarray]:
    return {
        name: np.full((8, 12, 3), value + index, dtype=np.uint8)
        for index, name in enumerate(robotwin_diagnostics.CAMERA_NAMES)
    }


def test_absolute_to_training_actions_preserves_grippers_and_deltas_joints():
    state = np.arange(14, dtype=np.float32)
    actions = np.stack((state + 1.0, state - 2.0))
    actions[:, 6] = (0.0, 1.0)
    actions[:, 13] = (1.0, 0.0)

    converted = robotwin_diagnostics.absolute_to_training_actions(actions, state)

    np.testing.assert_allclose(converted[:, robotwin_diagnostics.DELTA_JOINT_MASK], [[1.0] * 12, [-2.0] * 12])
    np.testing.assert_allclose(converted[:, 6], [0.0, 1.0])
    np.testing.assert_allclose(converted[:, 13], [1.0, 0.0])


def test_writer_records_shapes_actions_and_bounded_rgb(tmp_path):
    writer = robotwin_diagnostics.RoboTwinDiagnosticWriter(tmp_path, max_image_requests_per_episode=1)
    writer.start_episode()
    state = np.zeros(14, dtype=np.float32)
    actions = np.ones((3, 14), dtype=np.float32)

    first = writer.record_request(
        state=state,
        images=_images(),
        actions=actions,
        instruction="test task",
        action_chunk_size=3,
        executed_action_count=3,
        env_step_before=0,
        env_step_after=3,
    )
    second = writer.record_request(
        state=state,
        images=_images(30),
        actions=actions,
        instruction="test task",
        action_chunk_size=3,
        executed_action_count=2,
        env_step_before=3,
        env_step_after=5,
    )

    assert first["training_action_summary"]["shape"] == [3, 14]
    assert len(first["saved_images"]) == 3
    assert second["saved_images"] == {}
    rows = [json.loads(line) for line in writer.request_log_path.read_text().splitlines()]
    assert [row["request"] for row in rows] == [0, 1]
    assert rows[1]["executed_action_count"] == 2
    for path in first["saved_images"].values():
        with Image.open(path) as image:
            assert image.size == (12, 8)


def test_analyze_records_compares_training_space_to_reference(tmp_path):
    writer = robotwin_diagnostics.RoboTwinDiagnosticWriter(tmp_path, max_image_requests_per_episode=0)
    writer.start_episode()
    state = np.zeros(14, dtype=np.float32)
    actions = np.zeros((2, 14), dtype=np.float32)
    actions[1, 0] = 2.0
    actions[:, 6] = 0.5
    actions[:, 13] = 0.5
    record = writer.record_request(
        state=state,
        images=_images(),
        actions=actions,
        instruction="test task",
        action_chunk_size=2,
        executed_action_count=2,
        env_step_before=0,
        env_step_after=2,
    )
    reference = {
        "norm_stats": {
            "actions": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
            "state": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
        }
    }

    summary = robotwin_diagnostics.analyze_records([record], reference)

    assert summary["requests"] == 1
    assert summary["returned_actions"] == 2
    assert summary["executed_actions"] == 2
    assert summary["training_action_reference_check"]["above_count_per_dim"][0] == 1
    assert summary["gripper_outside_unit_interval"] == {"6": 0, "13": 0}
