from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).with_name("validate_hypothesis_c_scale_endpoint.py")
SPEC = importlib.util.spec_from_file_location("validate_hypothesis_c_scale_endpoint", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _fixture(root: Path, *, duplicate: bool = False) -> tuple[Path, Path]:
    result_root = root / "results"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    MODULE.SUITE_COUNTS = {"suite_a": 2, "suite_b": 1}
    for suite, count in MODULE.SUITE_COUNTS.items():
        run = result_root / f"run_{suite}_full"
        run.mkdir(parents=True)
        rows = [
            {
                "type": "episode",
                "suite": suite,
                "task_id": 0 if duplicate else task_id,
                "episode_idx": 0,
                "seed": 7,
                "success": task_id == 0,
                "error": None,
            }
            for task_id in range(count)
        ]
        for index in range(3):
            selected = rows[index::3]
            (run / f"shard{index}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8"
            )
        with (result_root / f"run_{suite}_full_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["table", "group", "episodes", "successes"])
            writer.writeheader()
            writer.writerow({"table": "overall", "group": "overall", "episodes": count, "successes": 1})
    return result_root, checkpoint


def test_validate_endpoint_accepts_exact_two_suite_manifest(tmp_path: Path) -> None:
    result_root, checkpoint = _fixture(tmp_path)
    payload = MODULE.validate_endpoint(
        result_root=result_root,
        run_prefix="run",
        expected_seed=7,
        fusion_scale=0.75,
        checkpoint=checkpoint,
    )
    assert payload["aggregate"]["episodes"] == 3
    assert payload["aggregate"]["successes"] == 2


def test_validate_endpoint_rejects_duplicate_key(tmp_path: Path) -> None:
    result_root, checkpoint = _fixture(tmp_path, duplicate=True)
    with pytest.raises(MODULE.EndpointValidationError, match="Duplicate key"):
        MODULE.validate_endpoint(
            result_root=result_root,
            run_prefix="run",
            expected_seed=7,
            fusion_scale=1.25,
            checkpoint=checkpoint,
        )
