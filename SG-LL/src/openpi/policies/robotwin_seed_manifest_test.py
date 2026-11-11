import hashlib
import json

import pytest

from openpi.policies.robotwin_seed_manifest import DISCOVERY_VETTED_QUALIFICATION_MODE
from openpi.policies.robotwin_seed_manifest import build_episode_seed_manifest
from openpi.policies.robotwin_seed_manifest import load_discovery_vetted_episode_schedule
from openpi.policies.robotwin_seed_manifest import load_episode_seed_manifest
from openpi.policies.robotwin_seed_manifest import load_explicit_episode_schedule
from openpi.policies.robotwin_seed_manifest import validate_explicit_manifest_preflight_mode


def _row(seed: int, *, error=None, task="adjust_bottle", task_config="demo_clean") -> dict:
    return {"task": task, "task_config": task_config, "seed": seed, "success": False, "error": error}


def test_manifest_round_trip_preserves_order_and_validates_metadata(tmp_path) -> None:
    manifest = build_episode_seed_manifest(
        [_row(100002), _row(100005)],
        source_results="/abs/results/episodes.jsonl",
        task="adjust_bottle",
        task_config="demo_clean",
        seed_arg=0,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_episode_seed_manifest(
        path,
        task="adjust_bottle",
        task_config="demo_clean",
        expected_test_num=2,
    ) == (100002, 100005)


def test_manifest_builder_rejects_duplicate_seed_and_episode_errors() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_episode_seed_manifest(
            [_row(1), _row(1)],
            source_results="rows.jsonl",
            task="adjust_bottle",
            task_config="demo_clean",
            seed_arg=0,
        )
    with pytest.raises(ValueError, match="runtime error"):
        build_episode_seed_manifest(
            [_row(1, error="Traceback")],
            source_results="rows.jsonl",
            task="adjust_bottle",
            task_config="demo_clean",
            seed_arg=0,
        )


def test_manifest_loader_rejects_wrong_task_and_count(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "adjust_bottle",
                "task_config": "demo_clean",
                "source_results": "rows.jsonl",
                "seed_arg": 0,
                "accepted_seeds": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task mismatch"):
        load_episode_seed_manifest(path, task="lift_pot", task_config="demo_clean", expected_test_num=1)
    with pytest.raises(ValueError, match="test_num"):
        load_episode_seed_manifest(path, task="adjust_bottle", task_config="demo_clean", expected_test_num=2)


def test_discovery_vetted_replay_requires_verified_manifest_qualification(tmp_path) -> None:
    manifest = build_episode_seed_manifest(
        [_row(100002), _row(100005)],
        source_results="rows.jsonl",
        task="adjust_bottle",
        task_config="demo_clean",
        seed_arg=0,
    )
    manifest["expert_preflight_qualification"] = {
        "mode": DISCOVERY_VETTED_QUALIFICATION_MODE,
        "episode_count": 2,
        "source_rows_verified": True,
        "instruction_schedule_sha256": hashlib.sha256(
            json.dumps(["put the bottle away", "put the bottle away"], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    manifest["instructions"] = ["put the bottle away", "put the bottle away"]
    path = tmp_path / "qualified_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_episode_seed_manifest(
        path,
        task="adjust_bottle",
        task_config="demo_clean",
        expected_test_num=2,
        require_discovery_vetted_qualification=True,
    ) == (100002, 100005)

    manifest["expert_preflight_qualification"]["source_rows_verified"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source_rows_verified"):
        load_episode_seed_manifest(
            path,
            task="adjust_bottle",
            task_config="demo_clean",
            expected_test_num=2,
            require_discovery_vetted_qualification=True,
        )

    manifest["expert_preflight_qualification"]["source_rows_verified"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_discovery_vetted_episode_schedule(
        path,
        task="adjust_bottle",
        task_config="demo_clean",
        expected_test_num=2,
    ) == ((100002, "put the bottle away"), (100005, "put the bottle away"))

    assert load_explicit_episode_schedule(
        path,
        task="adjust_bottle",
        task_config="demo_clean",
        expected_test_num=2,
        explicit_manifest_preflight_mode="discovery_vetted",
    ) == ((100002, 100005), ("put the bottle away", "put the bottle away"))

    assert load_explicit_episode_schedule(
        path,
        task="adjust_bottle",
        task_config="demo_clean",
        expected_test_num=2,
        explicit_manifest_preflight_mode="verify",
    ) == ((100002, 100005), None)


def test_explicit_manifest_preflight_mode_is_fail_closed() -> None:
    assert validate_explicit_manifest_preflight_mode("verify", has_explicit_manifest=False) == "verify"
    assert (
        validate_explicit_manifest_preflight_mode("discovery_vetted", has_explicit_manifest=True) == "discovery_vetted"
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        validate_explicit_manifest_preflight_mode("discovery_vetted", has_explicit_manifest=False)
    with pytest.raises(ValueError, match="must be one of"):
        validate_explicit_manifest_preflight_mode("invalid", has_explicit_manifest=True)
