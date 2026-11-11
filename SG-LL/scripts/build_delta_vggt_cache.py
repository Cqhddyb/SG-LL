"""Build a reusable LIBERO-plus VGGT feature cache for scheme3.

The cache stores frozen current/future VGGT teacher tokens, token masks,
fallback flags, horizon ids, and numeric sample identity fields.  It is an
offline artifact builder only; it does not modify pi0.5 training or inference.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any

from check_delta_vggt_inputs import as_device_tree
from check_delta_vggt_inputs import build_vision_token_mask
from check_delta_vggt_inputs import encode_vggt_hidden
from check_delta_vggt_inputs import load_vggt
from check_delta_vggt_inputs import select_device
from delta_vggt_cache_utils import FEATURE_CACHE_MANIFEST_SCHEMA
from delta_vggt_cache_utils import FEATURE_CACHE_SHARD_SCHEMA
from delta_vggt_cache_utils import torch_dtype_from_name
from delta_vggt_cache_utils import validate_feature_tensors
from delta_vggt_cache_utils import write_json
import jax
import torch

import openpi.models.model as _model
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.transforms as _transforms

LOGGER = logging.getLogger("build_delta_vggt_cache")

NUMERIC_METADATA_FIELDS = (
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
)


@dataclasses.dataclass(frozen=True)
class _PreserveCacheMetadataTransform:
    transform: _transforms.DataTransformFn

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        metadata = data.get("cache_metadata")
        output = self.transform(data)
        if metadata is not None and isinstance(output, dict) and "cache_metadata" not in output:
            output["cache_metadata"] = metadata
        return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_libero_plus_geo")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--shuffle-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--future-teacher-obs-offset",
        type=int,
        default=None,
        help=(
            "Explicitly override config.data.future_teacher_obs_offset for cache construction. "
            "Use this for timing ablations; the default config is unchanged when omitted."
        ),
    )
    parser.add_argument("--horizon-id", type=int, default=None)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--storage-dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--shard-size", type=int, default=16, help="Approximate number of samples per shard.")
    parser.add_argument("--no-load-vggt-weights", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to artifacts/delta_vggt_cache/<timestamp>.",
    )
    return parser.parse_args()


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(output_dir_arg: str | None, config_name: str) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts/delta_vggt_cache") / f"{config_name}_{timestamp}"
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    return output_dir


def _horizon_id(config: _config.TrainConfig, horizon_arg: int | None) -> int:
    if horizon_arg is not None:
        return int(horizon_arg)
    offset = config.data.future_teacher_obs_offset
    if offset is None:
        raise ValueError("Config has no future_teacher_obs_offset; pass --horizon-id explicitly.")
    return int(offset)


def _pooling_settings(config: _config.TrainConfig) -> tuple[str, bool, int]:
    pooling_func = "bilinear" if config.pooling_func is None else config.pooling_func
    use_vggt_pe = True if config.use_vggt_pe is None else bool(config.use_vggt_pe)
    vggt_layers_align = -1 if config.vggt_layers_align is None else int(config.vggt_layers_align)
    return pooling_func, use_vggt_pe, vggt_layers_align


def _extend_repack_with_cache_metadata(data_config: _config.DataConfig) -> list[_transforms.DataTransformFn]:
    transforms = []
    extended = False
    for transform in data_config.repack_transforms.inputs:
        if isinstance(transform, _transforms.RepackTransform):
            structure = copy.deepcopy(transform.structure)
            if "cache_metadata" in structure:
                raise ValueError("cache_metadata already exists in repack structure")
            structure["cache_metadata"] = {field: field for field in NUMERIC_METADATA_FIELDS}
            transforms.append(_transforms.RepackTransform(structure))
            extended = True
        else:
            transforms.append(transform)
    if not extended:
        raise ValueError("Expected a RepackTransform in data_config.repack_transforms")
    return transforms


def _create_cache_loader(config: _config.TrainConfig, *, shuffle: bool) -> _data.TorchDataLoader:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != "fake" and data_config.norm_stats is None:
        raise ValueError("Normalization stats are required to preserve the training preprocessing contract.")
    dataset = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transforms = [
        *_extend_repack_with_cache_metadata(data_config),
        *(_PreserveCacheMetadataTransform(transform) for transform in data_config.data_transforms.inputs),
        _PreserveCacheMetadataTransform(
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        ),
        *(_PreserveCacheMetadataTransform(transform) for transform in data_config.model_transforms.inputs),
    ]
    dataset = _data.TransformedDataset(dataset, transforms)
    return _data.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        shuffle=shuffle,
        num_batches=None,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="pytorch",
    )


def _validate_hidden(
    current_hidden: torch.Tensor,
    future_hidden: torch.Tensor,
    vision_mask: torch.Tensor,
    *,
    expected_dim: int,
) -> None:
    validate_feature_tensors(
        current_hidden,
        future_hidden,
        vision_mask,
        torch.zeros(current_hidden.shape[0], dtype=torch.long, device=current_hidden.device),
        torch.zeros(current_hidden.shape[0], dtype=torch.bool, device=current_hidden.device),
    )
    if current_hidden.shape[-1] != expected_dim:
        raise ValueError(f"VGGT dim {current_hidden.shape[-1]} does not match expected {expected_dim}")


def _tensor_to_cpu(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.detach().cpu().contiguous()


def _metadata_tensor(batch: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    metadata = batch.get("cache_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Batch is missing cache_metadata; cache repack failed")
    if key not in metadata:
        raise ValueError(f"Batch cache_metadata is missing {key!r}; available keys: {sorted(metadata)}")
    value = metadata[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device).view(-1)


def _collect_batch_tensors(
    batch_idx: int,
    batch: dict[str, Any],
    config: _config.TrainConfig,
    device: torch.device,
    vggt: torch.nn.Module,
    *,
    horizon_id: int,
    include_fallback: bool,
    storage_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    observation = _model.Observation.from_dict(batch)
    observation = jax.tree.map(lambda x: as_device_tree(x, device), observation)
    actions = batch["actions"].to(device)
    processed = _preprocessing.preprocess_observation_pytorch(observation, train=True, get_wo_aug=False)
    if not hasattr(processed, "teacher_images") or not hasattr(processed, "future_teacher_images"):
        raise ValueError("Processed observation does not contain teacher/future_teacher images")

    image_masks = list(processed.image_masks.values())
    image_padding_masks = list(processed.image_padding_mask.values())
    vision_mask, vision_metadata = build_vision_token_mask(image_masks, image_padding_masks)
    vision_mask = vision_mask.to(device=device).bool()

    fallback = getattr(processed, "future_teacher_fallback", None)
    fallback_bool = torch.zeros(vision_mask.shape[0], dtype=torch.bool, device=device)
    if fallback is not None:
        fallback_bool = fallback.to(device=device).bool().view(vision_mask.shape[0])
    row_keep = vision_mask.sum(dim=1) > 0
    if not include_fallback:
        row_keep &= ~fallback_bool

    pooling_func, use_vggt_pe, vggt_layers_align = _pooling_settings(config)
    teacher_images = list(processed.teacher_images.values())
    future_teacher_images = list(processed.future_teacher_images.values())
    current_hidden, current_meta = encode_vggt_hidden(
        teacher_images,
        vggt,
        vision_token_count=int(vision_mask.shape[1]),
        pooling_func=pooling_func,
        use_vggt_pe=use_vggt_pe,
        vggt_layers_align=vggt_layers_align,
    )
    future_hidden, future_meta = encode_vggt_hidden(
        future_teacher_images,
        vggt,
        vision_token_count=int(vision_mask.shape[1]),
        pooling_func=pooling_func,
        use_vggt_pe=use_vggt_pe,
        vggt_layers_align=vggt_layers_align,
    )
    expected_dim = int(config.geo_teacher_dim or config.vggt_dim)
    _validate_hidden(current_hidden, future_hidden, vision_mask, expected_dim=expected_dim)

    kept = int(row_keep.sum().detach().cpu())
    batch_summary = {
        "source_batch_idx": int(batch_idx),
        "input_batch_size": int(vision_mask.shape[0]),
        "kept_samples": kept,
        "fallback_samples": int(fallback_bool.sum().detach().cpu()),
        "empty_token_samples": int((vision_mask.sum(dim=1) <= 0).sum().detach().cpu()),
        "valid_tokens_per_input_sample": [int(x) for x in vision_mask.sum(dim=1).detach().cpu().tolist()],
        "vision_mask": vision_metadata,
        "current_hidden_shape": list(current_hidden.shape),
        "future_hidden_shape": list(future_hidden.shape),
        "current_meta": current_meta,
        "future_meta": future_meta,
    }
    if kept == 0:
        return None, batch_summary

    source_sample_idx = torch.arange(vision_mask.shape[0], dtype=torch.long, device=device)
    dataset_index = _metadata_tensor(batch, "index", device).long()
    raw_future_index = dataset_index + int(horizon_id)
    tensors = {
        "current_tokens": _tensor_to_cpu(current_hidden[row_keep], dtype=storage_dtype),
        "future_tokens": _tensor_to_cpu(future_hidden[row_keep], dtype=storage_dtype),
        "token_mask": _tensor_to_cpu(vision_mask[row_keep].bool()),
        "horizon_ids": _tensor_to_cpu(torch.full((kept,), horizon_id, dtype=torch.long, device=device)),
        "future_teacher_fallback": _tensor_to_cpu(fallback_bool[row_keep].bool()),
        "source_loader_batch_idx": _tensor_to_cpu(torch.full((kept,), batch_idx, dtype=torch.long, device=device)),
        "source_sample_idx": _tensor_to_cpu(source_sample_idx[row_keep]),
        "dataset_index": _tensor_to_cpu(dataset_index[row_keep]),
        "raw_future_index": _tensor_to_cpu(raw_future_index[row_keep]),
        "episode_index": _tensor_to_cpu(_metadata_tensor(batch, "episode_index", device).long()[row_keep]),
        "frame_index": _tensor_to_cpu(_metadata_tensor(batch, "frame_index", device).long()[row_keep]),
        "timestamp": _tensor_to_cpu(_metadata_tensor(batch, "timestamp", device).float()[row_keep]),
        "task_index": _tensor_to_cpu(_metadata_tensor(batch, "task_index", device).long()[row_keep]),
        "valid_token_counts": _tensor_to_cpu(vision_mask[row_keep].sum(dim=1).long()),
        "actions": _tensor_to_cpu(actions[row_keep].float()),
        "state": _tensor_to_cpu(observation.state[row_keep].float()),
    }
    validate_feature_tensors(
        tensors["current_tokens"],
        tensors["future_tokens"],
        tensors["token_mask"],
        tensors["horizon_ids"],
        tensors["future_teacher_fallback"],
    )
    return tensors, batch_summary


def _concat_buffer(buffer: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = buffer[0].keys()
    return {key: torch.cat([item[key] for item in buffer], dim=0) for key in keys}


def _write_shard(
    output_dir: Path,
    shard_idx: int,
    buffer: list[dict[str, torch.Tensor]],
    *,
    config_name: str,
) -> dict[str, Any]:
    tensors = _concat_buffer(buffer)
    sample_count = int(tensors["current_tokens"].shape[0])
    validate_feature_tensors(
        tensors["current_tokens"],
        tensors["future_tokens"],
        tensors["token_mask"],
        tensors["horizon_ids"],
        tensors["future_teacher_fallback"],
    )
    rel_path = Path("shards") / f"shard_{shard_idx:06d}.pt"
    payload = {
        "schema_version": FEATURE_CACHE_SHARD_SCHEMA,
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "config_name": config_name,
        "shard_index": int(shard_idx),
        "num_samples": sample_count,
        "tensors": tensors,
    }
    torch.save(payload, output_dir / rel_path)
    return {
        "path": str(rel_path),
        "num_samples": sample_count,
        "current_tokens_shape": list(tensors["current_tokens"].shape),
        "future_tokens_shape": list(tensors["future_tokens"].shape),
        "token_mask_shape": list(tensors["token_mask"].shape),
        "valid_tokens": int(tensors["token_mask"].sum().cpu()),
        "fallback_samples": int(tensors["future_teacher_fallback"].sum().cpu()),
    }


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    device = select_device(args.device)
    _set_seed(int(args.seed), device)
    config = _config.get_config(args.config_name)
    if args.future_teacher_obs_offset is not None:
        if not dataclasses.is_dataclass(config.data):
            raise TypeError("Config data object must be a dataclass to override future_teacher_obs_offset.")
        if not hasattr(config.data, "future_teacher_obs_offset"):
            raise ValueError(
                "--future-teacher-obs-offset was set, but this config data object has no "
                "future_teacher_obs_offset field."
            )
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, future_teacher_obs_offset=int(args.future_teacher_obs_offset)),
        )
    config = dataclasses.replace(
        config,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    horizon_id = _horizon_id(config, args.horizon_id)
    storage_dtype = torch_dtype_from_name(args.storage_dtype)
    output_dir = _resolve_output_dir(args.output_dir, config.name)
    write_json(output_dir / "args.json", vars(args))

    loader = _create_cache_loader(config, shuffle=bool(args.shuffle_data))
    vggt = load_vggt(config, device, load_weights=not args.no_load_vggt_weights)
    LOGGER.info("Loaded VGGT on %s (weights=%s)", device, not args.no_load_vggt_weights)

    shards: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    buffer: list[dict[str, torch.Tensor]] = []
    buffered_samples = 0
    total_samples = 0
    skipped_fallback = 0
    skipped_empty = 0
    first_shape: list[int] | None = None

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.num_batches):
            break
        LOGGER.info("Encoding cache batch %s/%s", batch_idx + 1, args.num_batches)
        tensors, batch_summary = _collect_batch_tensors(
            batch_idx,
            batch,
            config,
            device,
            vggt,
            horizon_id=horizon_id,
            include_fallback=bool(args.include_fallback),
            storage_dtype=storage_dtype,
        )
        batch_summaries.append(batch_summary)
        skipped_empty += int(batch_summary["empty_token_samples"])
        if not args.include_fallback:
            skipped_fallback += int(batch_summary["fallback_samples"])
        if tensors is None:
            continue
        if first_shape is None:
            first_shape = list(tensors["current_tokens"].shape[1:])
        sample_count = int(tensors["current_tokens"].shape[0])
        total_samples += sample_count
        buffer.append(tensors)
        buffered_samples += sample_count
        if buffered_samples >= int(args.shard_size):
            shard = _write_shard(output_dir, len(shards), buffer, config_name=config.name)
            LOGGER.info("Wrote %s with %s samples", shard["path"], shard["num_samples"])
            shards.append(shard)
            buffer = []
            buffered_samples = 0

    if buffer:
        shard = _write_shard(output_dir, len(shards), buffer, config_name=config.name)
        LOGGER.info("Wrote %s with %s samples", shard["path"], shard["num_samples"])
        shards.append(shard)

    del vggt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if total_samples <= 0:
        raise RuntimeError("No samples were written to the feature cache")

    pooling_func, use_vggt_pe, vggt_layers_align = _pooling_settings(config)
    manifest = {
        "schema_version": FEATURE_CACHE_MANIFEST_SCHEMA,
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "config_name": config.name,
        "repo_id": config.data.repo_id,
        "future_teacher_obs_offset": config.data.future_teacher_obs_offset,
        "horizon_id": int(horizon_id),
        "batch_size": int(args.batch_size),
        "num_batches_requested": int(args.num_batches),
        "num_batches_encoded": len(batch_summaries),
        "shuffle_data": bool(args.shuffle_data),
        "seed": int(args.seed),
        "include_fallback": bool(args.include_fallback),
        "storage_dtype": args.storage_dtype,
        "device": str(device),
        "vggt_weight_path": config.vggt_weight_path,
        "vggt_weights_loaded": not args.no_load_vggt_weights,
        "vggt_layers_align": vggt_layers_align,
        "pooling_func": pooling_func,
        "use_vggt_pe": use_vggt_pe,
        "teacher_dim": int(config.geo_teacher_dim or config.vggt_dim),
        "feature_shape_per_sample": first_shape,
        "metadata_fields": [
            *NUMERIC_METADATA_FIELDS,
            "raw_future_index",
            "source_loader_batch_idx",
            "source_sample_idx",
            "valid_token_counts",
            "actions",
            "state",
        ],
        "total_samples": int(total_samples),
        "skipped_fallback_samples": int(skipped_fallback),
        "skipped_empty_token_samples": int(skipped_empty),
        "shards": shards,
        "batch_summaries": batch_summaries,
    }
    write_json(output_dir / "manifest.json", manifest)
    print((output_dir / "manifest.json").read_text(encoding="utf-8"))
    LOGGER.info("Wrote cache manifest to %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
