#!/usr/bin/env python3
"""Paired success statistics for episode-level evaluation JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

EpisodeKey = tuple[Any, ...]


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)
    return expanded


def _read_episode_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _expand_paths(paths):
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type", "episode") != "episode":
                    continue
                row["_source_path"] = str(path)
                row["_source_line"] = line_no
                rows.append(row)
    if not rows:
        raise ValueError(f"No episode rows found in {paths}")
    return rows


def _is_robotwin_benchmark(benchmark: str) -> bool:
    return benchmark.strip().lower().replace("-", "") == "robotwin"


def episode_key(row: dict[str, Any], benchmark: str) -> EpisodeKey:
    if _is_robotwin_benchmark(benchmark):
        task = str(row.get("task", ""))
        task_config = str(row.get("task_config", ""))
        if not task or not task_config:
            raise ValueError(f"RoboTwin row is missing task/task_config: {row}")
        seed = int(row["seed"])
        if seed < 0:
            raise ValueError(f"RoboTwin seed must be non-negative: {row}")
        return (benchmark, task, task_config, seed)

    suite = str(row.get("suite", ""))
    if not suite:
        raise ValueError(f"Row is missing suite: {row}")
    task_id = int(row["task_id"])
    episode_id = int(row.get("episode_id", row.get("episode_idx", 0)))
    eval_seed = int(row.get("eval_seed", row.get("seed", 0)))
    return (benchmark, suite, task_id, episode_id, eval_seed)


def _index_rows(rows: list[dict[str, Any]], benchmark: str) -> tuple[dict[EpisodeKey, dict[str, Any]], int]:
    index: dict[EpisodeKey, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = episode_key(row, benchmark)
        if key in index:
            duplicates += 1
            continue
        index[key] = row
    return index, duplicates


def _error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("error") not in (None, "")]


def _load_robotwin_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read RoboTwin seed manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported RoboTwin seed manifest schema in {path}: {manifest.get('schema_version')!r}")
    task = manifest.get("task")
    task_config = manifest.get("task_config")
    seeds = manifest.get("accepted_seeds")
    if not isinstance(task, str) or not task or not isinstance(task_config, str) or not task_config:
        raise ValueError(f"RoboTwin seed manifest {path} must define non-empty task and task_config")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError(f"RoboTwin seed manifest {path} has invalid accepted_seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"RoboTwin seed manifest {path} has duplicate accepted_seeds")
    return manifest


def validate_robotwin_rows_against_manifest(rows: list[dict[str, Any]], *, manifest_path: Path, label: str) -> None:
    """Require one successful evaluator record for every manifest seed, in order."""
    manifest = _load_robotwin_manifest(manifest_path)
    expected_task = manifest["task"]
    expected_task_config = manifest["task_config"]
    expected_seeds = manifest["accepted_seeds"]

    if len(rows) != len(expected_seeds):
        raise ValueError(
            f"{label} has {len(rows)} episode rows, but manifest {manifest_path} requires {len(expected_seeds)}"
        )
    actual_seeds: list[int] = []
    for index, row in enumerate(rows):
        if row.get("task") != expected_task or row.get("task_config") != expected_task_config:
            raise ValueError(
                f"{label} row {index} task metadata does not match manifest {manifest_path}: "
                f"{row.get('task')!r}/{row.get('task_config')!r}"
            )
        if row.get("error") not in (None, ""):
            raise ValueError(f"{label} row {index} recorded evaluator error: {row['error']!r}")
        try:
            actual_seeds.append(int(row["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {index} has invalid seed: {row}") from exc
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"{label} seed order does not exactly match manifest {manifest_path}: "
            f"expected={expected_seeds}, actual={actual_seeds}"
        )


def _mcnemar_exact_pvalue(a_success_b_fail: int, a_fail_b_success: int) -> float | None:
    n = a_success_b_fail + a_fail_b_success
    if n == 0:
        return None
    k = min(a_success_b_fail, a_fail_b_success)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(k + 1)
    ]
    max_log = max(log_terms)
    lower_tail = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, 2.0 * lower_tail)


def _bootstrap_ci(diffs: list[int], *, samples: int, seed: int, alpha: float = 0.05) -> dict[str, float] | None:
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    means.sort()
    lo_idx = max(0, int((alpha / 2.0) * samples))
    hi_idx = min(samples - 1, int((1.0 - alpha / 2.0) * samples))
    return {
        "mean_diff": sum(diffs) / n,
        "ci_low": means[lo_idx],
        "ci_high": means[hi_idx],
        "samples": samples,
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "a_name",
        "b_name",
        "common_episodes",
        "a_successes",
        "b_successes",
        "a_success_rate",
        "b_success_rate",
        "paired_success_rate_diff",
        "a_success_b_fail",
        "a_fail_b_success",
        "both_success",
        "both_fail",
        "mcnemar_exact_pvalue",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "only_a_episodes",
        "only_b_episodes",
        "a_duplicate_keys",
        "b_duplicate_keys",
        "a_error_episodes",
        "b_error_episodes",
        "seed_manifest",
        "manifest_expected_episodes",
    ]
    row = {key: summary.get(key, "") for key in fieldnames}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _write_per_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "suite",
        "task_id",
        "episode_id",
        "eval_seed",
        "task_config",
        "seed",
        "a_success",
        "b_success",
        "diff",
        "a_task",
        "b_task",
        "a_source_path",
        "b_source_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-results", type=Path, nargs="+", required=True)
    parser.add_argument("--b-results", type=Path, nargs="+", required=True)
    parser.add_argument("--a-name", required=True)
    parser.add_argument("--b-name", required=True)
    parser.add_argument("--benchmark", default="LIBERO")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=None,
        help="RoboTwin manifest required for both result sets; validates task/config/seed order and error-free rows.",
    )
    parser.add_argument(
        "--require-no-errors",
        action="store_true",
        help="Reject result sets that contain a non-empty per-episode error field.",
    )
    parser.add_argument(
        "--require-exact-match",
        action="store_true",
        help="Reject duplicate or non-common episode keys instead of only reporting them.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-summary-csv", type=Path, required=True)
    parser.add_argument("--out-per-episode-csv", type=Path, default=None)
    args = parser.parse_args()

    a_rows = _read_episode_rows(args.a_results)
    b_rows = _read_episode_rows(args.b_results)
    if args.seed_manifest is not None:
        if not _is_robotwin_benchmark(args.benchmark):
            raise SystemExit("--seed-manifest is currently supported only with --benchmark RoboTwin.")
        try:
            validate_robotwin_rows_against_manifest(a_rows, manifest_path=args.seed_manifest, label=args.a_name)
            validate_robotwin_rows_against_manifest(b_rows, manifest_path=args.seed_manifest, label=args.b_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    a_error_rows = _error_rows(a_rows)
    b_error_rows = _error_rows(b_rows)
    if args.require_no_errors and (a_error_rows or b_error_rows):
        raise SystemExit(
            f"Per-episode errors are not allowed: {args.a_name}={len(a_error_rows)}, {args.b_name}={len(b_error_rows)}"
        )
    a_index, a_duplicates = _index_rows(a_rows, args.benchmark)
    b_index, b_duplicates = _index_rows(b_rows, args.benchmark)

    common_keys = sorted(set(a_index) & set(b_index))
    only_a = sorted(set(a_index) - set(b_index))
    only_b = sorted(set(b_index) - set(a_index))
    if not common_keys:
        raise SystemExit("No common episode keys found; check benchmark/suite/task_id/episode_id/seed alignment.")
    if args.require_exact_match and (a_duplicates or b_duplicates or only_a or only_b):
        raise SystemExit(
            "Exact paired comparison required, but result keys differ: "
            f"a_duplicates={a_duplicates}, b_duplicates={b_duplicates}, only_a={len(only_a)}, only_b={len(only_b)}"
        )

    per_episode_rows: list[dict[str, Any]] = []
    diffs: list[int] = []
    a_successes = 0
    b_successes = 0
    a_success_b_fail = 0
    a_fail_b_success = 0
    both_success = 0
    both_fail = 0
    for key in common_keys:
        a_row = a_index[key]
        b_row = b_index[key]
        a_success = bool(a_row.get("success", False))
        b_success = bool(b_row.get("success", False))
        a_successes += int(a_success)
        b_successes += int(b_success)
        diff = int(a_success) - int(b_success)
        diffs.append(diff)
        if a_success and not b_success:
            a_success_b_fail += 1
        elif (not a_success) and b_success:
            a_fail_b_success += 1
        elif a_success and b_success:
            both_success += 1
        else:
            both_fail += 1
        if _is_robotwin_benchmark(args.benchmark):
            benchmark, task, task_config, seed = key
            key_fields = {
                "benchmark": benchmark,
                "suite": "",
                "task_id": "",
                "episode_id": a_row.get("episode", ""),
                "eval_seed": seed,
                "task_config": task_config,
                "seed": seed,
                "a_task": task,
                "b_task": task,
            }
        else:
            benchmark, suite, task_id, episode_id, eval_seed = key
            key_fields = {
                "benchmark": benchmark,
                "suite": suite,
                "task_id": task_id,
                "episode_id": episode_id,
                "eval_seed": eval_seed,
                "task_config": "",
                "seed": "",
                "a_task": a_row.get("task", ""),
                "b_task": b_row.get("task", ""),
            }
        per_episode_rows.append(
            {
                **key_fields,
                "a_success": int(a_success),
                "b_success": int(b_success),
                "diff": diff,
                "a_source_path": a_row.get("_source_path", ""),
                "b_source_path": b_row.get("_source_path", ""),
            }
        )

    n = len(common_keys)
    bootstrap = _bootstrap_ci(diffs, samples=args.bootstrap_samples, seed=args.seed)
    summary = {
        "benchmark": args.benchmark,
        "a_name": args.a_name,
        "b_name": args.b_name,
        "common_episodes": n,
        "a_successes": a_successes,
        "b_successes": b_successes,
        "a_success_rate": a_successes / n,
        "b_success_rate": b_successes / n,
        "paired_success_rate_diff": (a_successes - b_successes) / n,
        "a_success_b_fail": a_success_b_fail,
        "a_fail_b_success": a_fail_b_success,
        "both_success": both_success,
        "both_fail": both_fail,
        "mcnemar_exact_pvalue": _mcnemar_exact_pvalue(a_success_b_fail, a_fail_b_success),
        "bootstrap_ci_low": None if bootstrap is None else bootstrap["ci_low"],
        "bootstrap_ci_high": None if bootstrap is None else bootstrap["ci_high"],
        "bootstrap": bootstrap,
        "only_a_episodes": len(only_a),
        "only_b_episodes": len(only_b),
        "a_duplicate_keys": a_duplicates,
        "b_duplicate_keys": b_duplicates,
        "a_error_episodes": len(a_error_rows),
        "b_error_episodes": len(b_error_rows),
        "seed_manifest": None if args.seed_manifest is None else str(args.seed_manifest),
        "manifest_expected_episodes": None
        if args.seed_manifest is None
        else len(_load_robotwin_manifest(args.seed_manifest)["accepted_seeds"]),
        "key_schema": ["benchmark", "task", "task_config", "seed"]
        if _is_robotwin_benchmark(args.benchmark)
        else ["benchmark", "suite", "task_id", "episode_id_or_episode_idx", "eval_seed_or_seed"],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_csv(args.out_summary_csv, summary)
    if args.out_per_episode_csv is not None:
        _write_per_episode_csv(args.out_per_episode_csv, per_episode_rows)

    print(f"Common episodes: {n}")
    print(f"{args.a_name}: {a_successes}/{n} = {a_successes / n:.4f}")
    print(f"{args.b_name}: {b_successes}/{n} = {b_successes / n:.4f}")
    print(f"Paired diff: {(a_successes - b_successes) / n:.4f}")
    print(f"McNemar exact p-value: {summary['mcnemar_exact_pvalue']}")
    print(f"Per-episode errors: {args.a_name}={len(a_error_rows)}, {args.b_name}={len(b_error_rows)}")
    print(f"Wrote JSON summary: {args.out_json}")
    print(f"Wrote CSV summary: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
