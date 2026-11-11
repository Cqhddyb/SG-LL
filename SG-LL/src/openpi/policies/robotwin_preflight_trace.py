"""Structured, opt-in tracing for RoboTwin expert-preflight reproducibility.

The normal RoboTwin evaluator imports this module only when the explicit
``robotwin_preflight_trace_path`` option is supplied.  The trace is designed
for protocol debugging: it records the state of the existing preflight
schedule without changing its acceptance decisions or policy execution.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import torch

_SCALAR_ENV_FIELDS = (
    "task_name",
    "ep_num",
    "model_name",
    "model_id",
    "table_z_bias",
    "crazy_random_light",
    "prohibited_area",
)
_SKIP_ACTOR_FIELDS = frozenset({"scene", "robot", "viewer", "engine", "renderer"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert bounded diagnostic values to a canonical JSON-compatible form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def rng_fingerprints() -> dict[str, Any]:
    """Return hashes for the NumPy, CPU Torch, CUDA Torch, and Python RNG states."""
    numpy_fingerprint = _sha256_bytes(pickle.dumps(np.random.get_state(), protocol=4))
    python_fingerprint = _sha256_bytes(pickle.dumps(random.getstate(), protocol=4))

    torch_cpu = _sha256_bytes(torch.get_rng_state().detach().cpu().numpy().tobytes())
    torch_cuda: list[str] = []
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            state = torch.cuda.get_rng_state(device_index).detach().cpu().numpy().tobytes()
            torch_cuda.append(_sha256_bytes(state))

    return {
        "numpy": numpy_fingerprint,
        "python_random": python_fingerprint,
        "torch_cpu": torch_cpu,
        "torch_cuda": torch_cuda,
    }


def _pose_summary(actor: Any) -> dict[str, Any] | None:
    get_pose = getattr(actor, "get_pose", None)
    if not callable(get_pose):
        return None
    try:
        pose = get_pose()
        return {
            "position": _json_value(np.asarray(pose.p, dtype=np.float64)),
            "quaternion": _json_value(np.asarray(pose.q, dtype=np.float64)),
        }
    except Exception as exc:  # The trace must never change evaluator control flow.
        return {"pose_error": f"{type(exc).__name__}: {exc}"}


def scene_snapshot(task_env: Any) -> dict[str, Any]:
    """Capture a small, task-agnostic scene summary before ``close_env``.

    Any environment member exposing ``get_pose`` is included except for the
    large simulator/controller handles.  For ``lift_pot`` this includes the
    kitchen pot pose and model id, which are enough to compare the initialized
    expert-preflight scene across automatic and explicit schedules.
    """
    scalar_fields = {
        name: _json_value(getattr(task_env, name)) for name in _SCALAR_ENV_FIELDS if hasattr(task_env, name)
    }
    actor_poses: dict[str, dict[str, Any]] = {}
    for name, value in sorted(vars(task_env).items()):
        if name in _SKIP_ACTOR_FIELDS:
            continue
        pose = _pose_summary(value)
        if pose is not None:
            actor_poses[name] = pose

    snapshot = {
        "fields": scalar_fields,
        "actors": actor_poses,
        "episode_info": _json_value(getattr(task_env, "info", {})),
    }
    encoded = json.dumps(snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"fingerprint": _sha256_bytes(encoded), "snapshot": snapshot}


def build_preflight_trace_record(
    *,
    mode: str,
    candidate_seed: int,
    now_ep_num: int,
    rng_before_setup: Mapping[str, Any],
    rng_after_setup: Mapping[str, Any] | None,
    rng_after_play_once: Mapping[str, Any] | None,
    scene_after_setup: Mapping[str, Any] | None,
    scene_after_play_once: Mapping[str, Any] | None,
    plan_success: bool | None,
    check_success: bool | None,
    exception: BaseException | None,
) -> dict[str, Any]:
    """Build one JSONL-safe trace record for the evaluator's actual preflight."""
    accepted = None if plan_success is None or check_success is None else bool(plan_success and check_success)
    return {
        "schema_version": 1,
        "mode": str(mode),
        "candidate_seed": int(candidate_seed),
        "now_ep_num": int(now_ep_num),
        "rng_before_setup": dict(rng_before_setup),
        "rng_after_setup": None if rng_after_setup is None else dict(rng_after_setup),
        "rng_after_play_once": None if rng_after_play_once is None else dict(rng_after_play_once),
        "scene_after_setup": None if scene_after_setup is None else dict(scene_after_setup),
        "scene_after_play_once": None if scene_after_play_once is None else dict(scene_after_play_once),
        "plan_success": plan_success,
        "check_success": check_success,
        "preflight_accepted": accepted,
        # Keep the legacy field for existing diagnostic readers; new analyses
        # compare the explicit before/after-play snapshots above.
        "scene": None if scene_after_play_once is None else dict(scene_after_play_once),
        "exception": None if exception is None else {"type": type(exception).__name__, "message": str(exception)},
    }


def append_preflight_trace_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one complete, flushed diagnostic record without touching evaluator results."""
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
