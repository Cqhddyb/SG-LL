from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.policies import robotwin_preflight_trace as trace


class _Pose:
    def __init__(self, position: list[float], quaternion: list[float]) -> None:
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray(quaternion, dtype=np.float64)


class _Actor:
    def __init__(self, position: list[float]) -> None:
        self.pose = _Pose(position, [1.0, 0.0, 0.0, 0.0])

    def get_pose(self) -> _Pose:
        return self.pose


class _Environment:
    task_name = "lift_pot"
    ep_num = 12
    model_name = "060_kitchenpot"
    model_id = 1
    table_z_bias = -0.01
    crazy_random_light = 0

    def __init__(self) -> None:
        self.prohibited_area = [[-0.3, -0.1, 0.3, 0.1]]
        self.pot = _Actor([0.1, 0.2, 0.3])
        self.robot = object()
        self.info = {"info": {"{A}": "060_kitchenpot/base1"}}


def test_scene_snapshot_tracks_actor_pose_and_ignores_controller_handles() -> None:
    environment = _Environment()

    first = trace.scene_snapshot(environment)
    environment.pot.pose = _Pose([0.1, 0.2, 0.4], [1.0, 0.0, 0.0, 0.0])
    second = trace.scene_snapshot(environment)

    assert first["snapshot"]["actors"]["pot"]["position"] == [0.1, 0.2, 0.3]
    assert "robot" not in first["snapshot"]["actors"]
    assert first["fingerprint"] != second["fingerprint"]


def test_trace_record_has_explicit_acceptance_and_serializable_rng_fields() -> None:
    record = trace.build_preflight_trace_record(
        mode="explicit_manifest",
        candidate_seed=100047,
        now_ep_num=12,
        rng_before_setup={"numpy": "before"},
        rng_after_setup={"numpy": "after_setup"},
        rng_after_play_once={"numpy": "after_play"},
        scene_after_setup=trace.scene_snapshot(_Environment()),
        scene_after_play_once=trace.scene_snapshot(_Environment()),
        plan_success=True,
        check_success=False,
        exception=None,
    )

    assert record["preflight_accepted"] is False
    assert record["exception"] is None
    assert record["scene_after_setup"]["snapshot"]["fields"]["model_id"] == 1
    assert record["scene_after_play_once"] == record["scene"]
    json.dumps(record, sort_keys=True)


def test_append_preflight_trace_record_writes_complete_jsonl_row(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "trace.jsonl"
    record = trace.build_preflight_trace_record(
        mode="automatic_discovery",
        candidate_seed=100047,
        now_ep_num=12,
        rng_before_setup={},
        rng_after_setup=None,
        rng_after_play_once=None,
        scene_after_setup=None,
        scene_after_play_once=None,
        plan_success=None,
        check_success=None,
        exception=RuntimeError("preflight failure"),
    )

    trace.append_preflight_trace_record(output, record)

    assert json.loads(output.read_text(encoding="utf-8")) == record
