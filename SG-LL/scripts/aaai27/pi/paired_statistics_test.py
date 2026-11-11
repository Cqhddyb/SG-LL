"""Unit tests for LIBERO/RoboTwin paired success statistics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SCRIPT_PATH = Path(__file__).with_name("paired_statistics.py")
_SPEC = importlib.util.spec_from_file_location("paired_statistics_under_test", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
paired_statistics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = paired_statistics
_SPEC.loader.exec_module(paired_statistics)


def _robotwin_row(seed: int, *, success: bool, error: str | None = None) -> dict[str, object]:
    return {
        "task": "adjust_bottle",
        "task_config": "demo_clean",
        "episode": seed - 100000,
        "seed": seed,
        "success": success,
        "steps": 400,
        "error": error,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "adjust_bottle",
                "task_config": "demo_clean",
                "accepted_seeds": [100000, 100002],
            }
        ),
        encoding="utf-8",
    )


def test_robotwin_key_uses_task_config_and_seed() -> None:
    row = _robotwin_row(100000, success=True)
    assert paired_statistics.episode_key(row, "RoboTwin") == (
        "RoboTwin",
        "adjust_bottle",
        "demo_clean",
        100000,
    )


def test_robotwin_manifest_requires_order_and_error_free_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    reversed_rows = [_robotwin_row(100002, success=False), _robotwin_row(100000, success=True)]
    with pytest.raises(ValueError, match="seed order"):
        paired_statistics.validate_robotwin_rows_against_manifest(
            reversed_rows, manifest_path=manifest, label="route_a"
        )

    error_rows = [
        _robotwin_row(100000, success=False, error="renderer failed"),
        _robotwin_row(100002, success=True),
    ]
    with pytest.raises(ValueError, match="recorded evaluator error"):
        paired_statistics.validate_robotwin_rows_against_manifest(error_rows, manifest_path=manifest, label="baseline")


def test_robotwin_cli_writes_exact_paired_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    a_results = tmp_path / "route_a.jsonl"
    b_results = tmp_path / "baseline.jsonl"
    out_json = tmp_path / "summary.json"
    out_csv = tmp_path / "summary.csv"
    out_per_episode = tmp_path / "per_episode.csv"
    _write_manifest(manifest)
    _write_jsonl(a_results, [_robotwin_row(100000, success=True), _robotwin_row(100002, success=False)])
    _write_jsonl(b_results, [_robotwin_row(100000, success=False), _robotwin_row(100002, success=False)])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT_PATH),
            "--a-results",
            str(a_results),
            "--b-results",
            str(b_results),
            "--a-name",
            "route_a",
            "--b-name",
            "baseline",
            "--benchmark",
            "RoboTwin",
            "--seed-manifest",
            str(manifest),
            "--require-no-errors",
            "--require-exact-match",
            "--bootstrap-samples",
            "100",
            "--out-json",
            str(out_json),
            "--out-summary-csv",
            str(out_csv),
            "--out-per-episode-csv",
            str(out_per_episode),
        ],
    )
    paired_statistics.main()

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["key_schema"] == ["benchmark", "task", "task_config", "seed"]
    assert summary["common_episodes"] == 2
    assert summary["a_successes"] == 1
    assert summary["b_successes"] == 0
    assert summary["paired_success_rate_diff"] == 0.5
    assert summary["a_error_episodes"] == 0
    assert summary["b_error_episodes"] == 0
    assert summary["manifest_expected_episodes"] == 2
    assert out_csv.exists()
    assert out_per_episode.exists()
