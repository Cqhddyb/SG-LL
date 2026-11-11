#!/usr/bin/env python3
"""Summarize LIBERO / LIBERO-plus episode JSONL results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type", "episode") == "episode":
                rows.append(row)
    return rows


def _expand_result_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)
    return expanded


def _load_task_classification(path: Path, suite: str) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if suite not in data:
        raise ValueError(f"Suite {suite!r} not found in {path}. Available suites: {sorted(data)}")
    return {int(item["id"]): item for item in data[suite]}


def _rate(successes: int, episodes: int) -> float:
    return successes / episodes if episodes else 0.0


def _add(agg: dict[str, dict[str, int]], key: str, success: bool) -> None:
    agg[key]["episodes"] += 1
    agg[key]["successes"] += int(success)


def _write_csv(path: Path, table_rows: Iterable[dict[str, Any]]) -> None:
    rows = list(table_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["table", "group", "episodes", "successes", "success_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _print_table(title: str, agg: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    print(f"\n{title}")
    print(f"{'Group':36s} {'Succ':>6s} {'Eps':>6s} {'Rate':>8s}")
    print("-" * 62)
    rows = []
    for group, val in sorted(agg.items(), key=lambda item: (-_rate(item[1]["successes"], item[1]["episodes"]), item[0])):
        episodes = val["episodes"]
        successes = val["successes"]
        success_rate = _rate(successes, episodes)
        print(f"{group[:36]:36s} {successes:6d} {episodes:6d} {success_rate * 100:7.2f}%")
        rows.append(
            {
                "table": title,
                "group": group,
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--task-classification", type=Path, required=True)
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument("--out-summary-csv", type=Path, default=None)
    args = parser.parse_args()

    result_paths = _expand_result_paths(args.results_jsonl)
    rows = []
    for path in result_paths:
        rows.extend(_read_jsonl(path))
    if not rows:
        raise SystemExit(f"No episode rows found in {result_paths}")

    suite = args.suite or str(rows[0].get("suite", ""))
    if not suite:
        raise SystemExit("Could not infer suite from JSONL; pass --suite explicitly.")
    task_meta = _load_task_classification(args.task_classification, suite)

    overall = defaultdict(lambda: {"episodes": 0, "successes": 0})
    by_category = defaultdict(lambda: {"episodes": 0, "successes": 0})
    by_difficulty = defaultdict(lambda: {"episodes": 0, "successes": 0})
    by_degradation = defaultdict(lambda: {"episodes": 0, "successes": 0})
    by_category_degradation = defaultdict(lambda: {"episodes": 0, "successes": 0})

    missing_classification = 0
    for row in rows:
        success = bool(row.get("success", False))
        classification_task_id = int(row.get("classification_task_id", int(row.get("task_id", -1)) + 1))
        meta = task_meta.get(classification_task_id)
        if meta is None:
            missing_classification += 1
            category = "unknown"
            difficulty = "unknown"
        else:
            category = str(meta.get("category", "unknown"))
            difficulty = str(meta.get("difficulty_level", "unknown"))

        degradation_env = row.get("degradation", {}) or {}
        degradation = str(degradation_env.get("LIBERO_IMAGE_DEGRADATION", "") or "clean")

        _add(overall, "overall", success)
        _add(by_category, category, success)
        _add(by_difficulty, difficulty, success)
        _add(by_degradation, degradation, success)
        _add(by_category_degradation, f"{category}|{degradation}", success)

    print("Results:")
    for path in result_paths:
        print(f"  {path}")
    print(f"Suite: {suite}")
    print(f"Episodes: {len(rows)}")
    if missing_classification:
        print(f"[warn] {missing_classification} rows did not match task classification.")

    output_rows = []
    output_rows.extend(_print_table("overall", overall))
    output_rows.extend(_print_table("category", by_category))
    output_rows.extend(_print_table("difficulty", by_difficulty))
    output_rows.extend(_print_table("degradation", by_degradation))
    output_rows.extend(_print_table("category|degradation", by_category_degradation))

    if args.out_summary_csv:
        _write_csv(args.out_summary_csv, output_rows)
        print(f"\nWrote summary CSV to: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
