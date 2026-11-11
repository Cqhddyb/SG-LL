"""Utilities for scheme3 VGGT feature caches."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

FEATURE_CACHE_MANIFEST_SCHEMA = "scheme3_vggt_feature_cache_manifest_v1"
FEATURE_CACHE_SHARD_SCHEMA = "scheme3_vggt_feature_cache_shard_v1"


@dataclasses.dataclass(frozen=True)
class FeatureCacheTensors:
    """Loaded feature cache tensors.

    Shapes:
    - current_tokens: [S, N_vision, D_teacher]
    - future_tokens: [S, N_vision, D_teacher]
    - token_mask: [S, N_vision]
    - horizon_ids: [S]
    - future_teacher_fallback: [S]
    """

    current_tokens: torch.Tensor
    future_tokens: torch.Tensor
    token_mask: torch.Tensor
    horizon_ids: torch.Tensor
    future_teacher_fallback: torch.Tensor
    metadata: dict[str, torch.Tensor]
    manifest: dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported storage dtype {name!r}; choose one of {sorted(dtype_map)}")
    return dtype_map[name]


def validate_feature_tensors(
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    horizon_ids: torch.Tensor,
    future_teacher_fallback: torch.Tensor,
) -> None:
    if current_tokens.ndim != 3 or future_tokens.ndim != 3:
        raise ValueError(
            f"current/future tokens must be rank 3, got {tuple(current_tokens.shape)} and {tuple(future_tokens.shape)}"
        )
    if current_tokens.shape != future_tokens.shape:
        raise ValueError(
            f"current/future token shapes differ: {tuple(current_tokens.shape)} vs {tuple(future_tokens.shape)}"
        )
    sample_count, token_count, _teacher_dim = current_tokens.shape
    if token_mask.shape != (sample_count, token_count):
        raise ValueError(f"token_mask must have shape {(sample_count, token_count)}, got {tuple(token_mask.shape)}")
    if horizon_ids.shape != (sample_count,):
        raise ValueError(f"horizon_ids must have shape {(sample_count,)}, got {tuple(horizon_ids.shape)}")
    if future_teacher_fallback.shape != (sample_count,):
        raise ValueError(
            f"future_teacher_fallback must have shape {(sample_count,)}, got {tuple(future_teacher_fallback.shape)}"
        )
    if int(token_mask.bool().sum().cpu()) <= 0:
        raise ValueError("token_mask has no valid tokens")


def load_manifest(cache_dir: Path | str) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    manifest = read_json(cache_dir / "manifest.json")
    if manifest.get("schema_version") != FEATURE_CACHE_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unexpected manifest schema {manifest.get('schema_version')!r}; expected {FEATURE_CACHE_MANIFEST_SCHEMA!r}"
        )
    return manifest


def shard_paths(cache_dir: Path | str, manifest: dict[str, Any]) -> list[Path]:
    cache_dir = Path(cache_dir)
    paths = []
    for shard in manifest.get("shards", []):
        rel_path = shard.get("path")
        if not rel_path:
            raise ValueError(f"Shard entry missing path: {shard}")
        paths.append(cache_dir / rel_path)
    if not paths:
        raise ValueError(f"No shards found in manifest under {cache_dir}")
    return paths


def load_shard(path: Path | str, *, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != FEATURE_CACHE_SHARD_SCHEMA:
        raise ValueError(
            f"Unexpected shard schema {payload.get('schema_version')!r}; expected {FEATURE_CACHE_SHARD_SCHEMA!r}"
        )
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError(f"Shard {path} is missing tensor payload")
    validate_feature_tensors(
        tensors["current_tokens"],
        tensors["future_tokens"],
        tensors["token_mask"],
        tensors["horizon_ids"],
        tensors["future_teacher_fallback"],
    )
    return payload


def load_feature_cache(
    cache_dir: Path | str,
    *,
    device: torch.device,
    max_samples: int | None = None,
    drop_fallback: bool = True,
) -> FeatureCacheTensors:
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    tensor_chunks: dict[str, list[torch.Tensor]] = {
        "current_tokens": [],
        "future_tokens": [],
        "token_mask": [],
        "horizon_ids": [],
        "future_teacher_fallback": [],
    }
    metadata_chunks: dict[str, list[torch.Tensor]] = {}

    loaded = 0
    for path in shard_paths(cache_dir, manifest):
        payload = load_shard(path, map_location="cpu")
        tensors = payload["tensors"]
        sample_count = int(tensors["current_tokens"].shape[0])
        take_count = sample_count
        if max_samples is not None:
            remaining = int(max_samples) - loaded
            if remaining <= 0:
                break
            take_count = min(take_count, remaining)

        row_mask = torch.ones(take_count, dtype=torch.bool)
        if drop_fallback:
            row_mask &= ~tensors["future_teacher_fallback"][:take_count].bool()
        if int(row_mask.sum()) == 0:
            loaded += take_count
            continue

        for key, chunks in tensor_chunks.items():
            chunks.append(tensors[key][:take_count][row_mask])
        for key, value in tensors.items():
            if key in tensor_chunks or not isinstance(value, torch.Tensor):
                continue
            if value.shape[:1] != (sample_count,):
                continue
            metadata_chunks.setdefault(key, []).append(value[:take_count][row_mask])
        loaded += take_count

    if not tensor_chunks["current_tokens"]:
        raise ValueError(f"No usable samples loaded from cache {cache_dir}")

    current_tokens = torch.cat(tensor_chunks["current_tokens"], dim=0).to(device=device)
    future_tokens = torch.cat(tensor_chunks["future_tokens"], dim=0).to(device=device)
    token_mask = torch.cat(tensor_chunks["token_mask"], dim=0).to(device=device).bool()
    horizon_ids = torch.cat(tensor_chunks["horizon_ids"], dim=0).to(device=device).long()
    fallback = torch.cat(tensor_chunks["future_teacher_fallback"], dim=0).to(device=device).bool()
    validate_feature_tensors(current_tokens, future_tokens, token_mask, horizon_ids, fallback)

    metadata = {key: torch.cat(chunks, dim=0).to(device=device) for key, chunks in metadata_chunks.items() if chunks}
    return FeatureCacheTensors(
        current_tokens=current_tokens,
        future_tokens=future_tokens,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        future_teacher_fallback=fallback,
        metadata=metadata,
        manifest=manifest,
    )


def feature_cache_shape_summary(features: FeatureCacheTensors) -> dict[str, Any]:
    return {
        "current_tokens": list(features.current_tokens.shape),
        "future_tokens": list(features.future_tokens.shape),
        "token_mask": list(features.token_mask.shape),
        "horizon_ids": list(features.horizon_ids.shape),
        "future_teacher_fallback": list(features.future_teacher_fallback.shape),
        "valid_tokens": int(features.token_mask.sum().detach().cpu()),
        "fallback_samples": int(features.future_teacher_fallback.sum().detach().cpu()),
    }
