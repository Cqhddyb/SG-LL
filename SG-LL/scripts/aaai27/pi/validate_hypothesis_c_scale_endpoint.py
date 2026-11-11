#!/usr/bin/env python3
"""Strictly validate one two-suite Hypothesis-C fusion-scale endpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SUITE_COUNTS = {"libero_spatial": 2402, "libero_10": 2519}


class EndpointValidationError(RuntimeError):
    """Raised when a fusion-scale endpoint violates its fixed manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    overall = [row for row in rows if row.get("table") == "overall" and row.get("group") == "overall"]
    if len(overall) != 1:
        raise EndpointValidationError(f"Expected one overall row in {path}, found {len(overall)}")
    return int(overall[0]["episodes"]), int(overall[0]["successes"])


def validate_endpoint(
    *,
    result_root: Path,
    run_prefix: str,
    expected_seed: int,
    fusion_scale: float,
    checkpoint: Path,
) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    all_keys: set[tuple[str, int, int, int]] = set()
    total_successes = 0
    for suite, expected_count in SUITE_COUNTS.items():
        run_name = f"{run_prefix}_{suite}_full"
        result_dir = result_root / run_name
        summary_path = result_root / f"{run_name}_summary.csv"
        shards = sorted(result_dir.glob("*.jsonl"))
        if len(shards) != 3:
            raise EndpointValidationError(f"{suite}: expected three shards, found {len(shards)}")
        successes = errors = rows = 0
        suite_keys: set[tuple[str, int, int, int]] = set()
        for shard in shards:
            with shard.open(encoding="utf-8") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise EndpointValidationError(f"Invalid JSONL at {shard}:{line_number}") from exc
                    if row.get("type", "episode") != "episode":
                        continue
                    if str(row.get("suite")) != suite or not isinstance(row.get("success"), bool):
                        raise EndpointValidationError(f"Malformed row at {shard}:{line_number}")
                    key = (
                        suite,
                        int(row["task_id"]),
                        int(row.get("episode_id", row.get("episode_idx"))),
                        int(row.get("eval_seed", row.get("seed"))),
                    )
                    if key in suite_keys:
                        raise EndpointValidationError(f"Duplicate key {key}")
                    suite_keys.add(key)
                    rows += 1
                    successes += int(row["success"])
                    errors += int(row.get("error") not in (None, ""))
        expected_keys = {(suite, task_id, 0, expected_seed) for task_id in range(expected_count)}
        if suite_keys != expected_keys or rows != expected_count or errors:
            raise EndpointValidationError(
                f"{suite}: manifest mismatch rows={rows}, errors={errors}, "
                f"missing={len(expected_keys - suite_keys)}, extra={len(suite_keys - expected_keys)}"
            )
        summary_rows, summary_successes = _summary(summary_path)
        if (summary_rows, summary_successes) != (rows, successes):
            raise EndpointValidationError(f"{suite}: summary/raw mismatch")
        all_keys.update(suite_keys)
        total_successes += successes
        suites[suite] = {
            "episodes": rows,
            "successes": successes,
            "success_rate": successes / rows,
            "errors": 0,
            "duplicate_keys": 0,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "jsonl_shards": 3,
            "result_dir": str(result_dir),
            "summary_csv": str(summary_path),
        }
    expected_total = sum(SUITE_COUNTS.values())
    if len(all_keys) != expected_total:
        raise EndpointValidationError(f"Expected {expected_total} unique keys, found {len(all_keys)}")
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise EndpointValidationError(f"Checkpoint model is missing: {model_path}")
    return {
        "status": "CONFIRMED_STRICT_VALIDATION",
        "fusion_scale": fusion_scale,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": _sha256(model_path),
        "eval_seed": expected_seed,
        "workers": 3,
        "trials_per_task": 1,
        "future_offset": 10,
        "video_policy": "disabled",
        "suites": suites,
        "aggregate": {
            "episodes": expected_total,
            "successes": total_successes,
            "success_rate": total_successes / expected_total,
            "errors": 0,
            "duplicate_keys": 0,
            "missing_keys": 0,
            "unexpected_keys": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--expected-seed", type=int, default=7)
    parser.add_argument("--fusion-scale", type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_endpoint(
        result_root=args.result_root,
        run_prefix=args.run_prefix,
        expected_seed=args.expected_seed,
        fusion_scale=args.fusion_scale,
        checkpoint=args.checkpoint,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
