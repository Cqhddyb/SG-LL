"""Validation and construction helpers for fixed RoboTwin episode seed manifests."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPLICIT_MANIFEST_PREFLIGHT_MODES = frozenset({"verify", "discovery_vetted"})
DISCOVERY_VETTED_QUALIFICATION_MODE = "automatic_expert_preflight"


def build_episode_seed_manifest(
    episode_rows: Iterable[dict[str, Any]],
    *,
    source_results: str,
    task: str,
    task_config: str,
    seed_arg: int,
) -> dict[str, Any]:
    """Build a formal manifest from finalized, successful-runtime episode rows.

    A seed manifest fixes the scene and instruction schedule for a paired
    comparison. It accepts successes and failures, but never episode rows with
    a runtime error or ambiguous task metadata.
    """
    accepted_seeds: list[int] = []
    seen_seeds: set[int] = set()
    for row_index, row in enumerate(episode_rows):
        if row.get("task") != task:
            raise ValueError(f"row {row_index} task mismatch: expected {task!r}, got {row.get('task')!r}")
        if row.get("task_config") != task_config:
            raise ValueError(
                f"row {row_index} task_config mismatch: expected {task_config!r}, got {row.get('task_config')!r}"
            )
        if row.get("error") is not None:
            raise ValueError(f"row {row_index} has a runtime error and cannot enter a fixed manifest")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"row {row_index} has invalid non-negative integer seed: {seed!r}")
        if seed in seen_seeds:
            raise ValueError(f"duplicate accepted seed {seed} at row {row_index}")
        seen_seeds.add(seed)
        accepted_seeds.append(seed)

    if not accepted_seeds:
        raise ValueError("cannot build an episode seed manifest from zero finalized rows")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "task_config": task_config,
        "source_results": source_results,
        "seed_arg": seed_arg,
        "accepted_seeds": accepted_seeds,
    }


def load_discovery_vetted_episode_schedule(
    path: str | Path,
    *,
    task: str,
    task_config: str,
    expected_test_num: int,
) -> tuple[tuple[int, str], ...]:
    """Load a qualified fixed `(seed, instruction)` schedule for policy replay.

    The expert rollout that qualifies a discovery seed also determines the
    instruction template. A replay that deliberately avoids a second expert
    rollout must reuse that instruction rather than drawing a new one.
    """

    seeds = load_episode_seed_manifest(
        path,
        task=task,
        task_config=task_config,
        expected_test_num=expected_test_num,
        require_discovery_vetted_qualification=True,
    )
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    instructions = raw.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != len(seeds):
        raise ValueError("discovery-vetted manifest instructions must match accepted_seeds")
    if any(not isinstance(instruction, str) or not instruction for instruction in instructions):
        raise ValueError("discovery-vetted manifest instructions must contain non-empty strings")

    qualification = raw["expert_preflight_qualification"]
    expected_instruction_sha256 = qualification.get("instruction_schedule_sha256")
    encoded = json.dumps(instructions, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    actual_instruction_sha256 = __import__("hashlib").sha256(encoded).hexdigest()
    if expected_instruction_sha256 != actual_instruction_sha256:
        raise ValueError("discovery-vetted manifest instruction schedule checksum mismatch")
    return tuple(zip(seeds, instructions, strict=True))


def load_explicit_episode_schedule(
    path: str | Path,
    *,
    task: str,
    task_config: str,
    expected_test_num: int,
    explicit_manifest_preflight_mode: str,
) -> tuple[tuple[int, ...], tuple[str, ...] | None]:
    """Resolve a validated explicit RoboTwin replay schedule.

    ``verify`` preserves the historical evaluator path: it uses only the
    manifest seeds and re-runs standard expert preflight, which reconstructs
    an instruction for each replayed seed. ``discovery_vetted`` is an opt-in
    paired protocol for an immutable, expert-qualified manifest. It reuses
    the discovery instruction verbatim because a second expert rollout is
    intentionally not part of the replay.
    """

    mode = validate_explicit_manifest_preflight_mode(
        explicit_manifest_preflight_mode,
        has_explicit_manifest=True,
    )
    if mode == "verify":
        return (
            load_episode_seed_manifest(
                path,
                task=task,
                task_config=task_config,
                expected_test_num=expected_test_num,
            ),
            None,
        )

    vetted_schedule = load_discovery_vetted_episode_schedule(
        path,
        task=task,
        task_config=task_config,
        expected_test_num=expected_test_num,
    )
    return (
        tuple(seed for seed, _ in vetted_schedule),
        tuple(instruction for _, instruction in vetted_schedule),
    )


def load_episode_seed_manifest(
    path: str | Path,
    *,
    task: str,
    task_config: str,
    expected_test_num: int,
    require_discovery_vetted_qualification: bool = False,
) -> tuple[int, ...]:
    """Load and validate an explicit seed manifest for one formal evaluation."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"episode seed manifest does not exist: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"episode seed manifest is invalid JSON: {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"episode seed manifest root must be an object: {manifest_path}")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported episode seed manifest schema_version={raw.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if raw.get("task") != task:
        raise ValueError(f"manifest task mismatch: expected {task!r}, got {raw.get('task')!r}")
    if raw.get("task_config") != task_config:
        raise ValueError(f"manifest task_config mismatch: expected {task_config!r}, got {raw.get('task_config')!r}")
    if not isinstance(raw.get("source_results"), str) or not raw["source_results"]:
        raise ValueError("manifest source_results must be a non-empty string")
    if isinstance(raw.get("seed_arg"), bool) or not isinstance(raw.get("seed_arg"), int):
        raise ValueError("manifest seed_arg must be an integer")
    seeds = raw.get("accepted_seeds")
    if not isinstance(seeds, list):
        raise ValueError("manifest accepted_seeds must be a list")
    if len(seeds) != expected_test_num:
        raise ValueError(
            f"manifest has {len(seeds)} accepted seeds, but evaluation requested test_num={expected_test_num}"
        )
    if not seeds:
        raise ValueError("manifest accepted_seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("manifest accepted_seeds must contain only non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("manifest accepted_seeds must be unique")
    if require_discovery_vetted_qualification:
        qualification = raw.get("expert_preflight_qualification")
        if not isinstance(qualification, dict):
            raise ValueError("manifest is missing expert_preflight_qualification for discovery-vetted replay")
        if qualification.get("mode") != DISCOVERY_VETTED_QUALIFICATION_MODE:
            raise ValueError("manifest expert_preflight_qualification is not from automatic expert preflight")
        if qualification.get("episode_count") != len(seeds):
            raise ValueError("manifest expert_preflight_qualification episode_count does not match accepted_seeds")
        if qualification.get("source_rows_verified") is not True:
            raise ValueError("manifest expert_preflight_qualification source_rows_verified must be true")
    return tuple(seeds)


def validate_explicit_manifest_preflight_mode(mode: object, *, has_explicit_manifest: bool) -> str:
    """Validate the evaluator's explicit-manifest expert-preflight policy.

    ``verify`` preserves RoboTwin's historical behavior by executing an expert
    rollout again before every policy episode. ``discovery_vetted`` is only
    valid for an immutable manifest whose discovery rows were qualified by the
    standard expert rollout. It avoids treating a nondeterministic second
    expert rollout as a requirement for replaying an already frozen policy
    episode schedule.
    """

    if not isinstance(mode, str) or mode not in EXPLICIT_MANIFEST_PREFLIGHT_MODES:
        raise ValueError(
            "robotwin_explicit_manifest_preflight_mode must be one of "
            f"{sorted(EXPLICIT_MANIFEST_PREFLIGHT_MODES)}, got {mode!r}"
        )
    if mode == "discovery_vetted" and not has_explicit_manifest:
        raise ValueError(
            "robotwin_explicit_manifest_preflight_mode='discovery_vetted' requires an explicit episode manifest"
        )
    return mode
