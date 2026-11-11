"""Augment a VGGT feature cache with frozen pi0.5 language summary features.

The script replays the source cache dataloader order and writes:

- language_features: [S, D_pi]
- optionally, language_token_features: [S, L_language, D_pi]
- optionally, language_token_mask: [S, L_language]

It does not rebuild VGGT features and does not modify the source cache.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from build_delta_vggt_cache import _create_cache_loader
from build_delta_vggt_cache import _metadata_tensor
from check_delta_vggt_inputs import as_device_tree
from check_delta_vggt_inputs import build_vision_token_mask
from check_delta_vggt_inputs import select_device
from delta_vggt_cache_utils import FEATURE_CACHE_MANIFEST_SCHEMA
from delta_vggt_cache_utils import load_shard
from delta_vggt_cache_utils import read_json
from delta_vggt_cache_utils import shard_paths
from delta_vggt_cache_utils import validate_feature_tensors
from delta_vggt_cache_utils import write_json
import jax
import safetensors.torch
import torch

import openpi.models.model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.training.config as _config

LOGGER = logging.getLogger("add_language_features_to_delta_cache")

DEFAULT_INPUT_CACHE = "artifacts/delta_vggt_cache/s3_m4_h10_n2048_seed31_cache_b2_target2048_shuffle_seed31"
DEFAULT_POLICY_CHECKPOINT = "checkpoints/pi05_libero_official_torch"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache-dir", default=DEFAULT_INPUT_CACHE)
    parser.add_argument("--output-cache-dir", default=None)
    parser.add_argument("--config-name", default=None, help="Defaults to the input cache manifest config_name.")
    parser.add_argument("--policy-checkpoint-dir", default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the input cache manifest seed.")
    parser.add_argument("--vla-layer", type=int, default=None, help="Defaults to TrainConfig.vla_layers_align or 12.")
    parser.add_argument(
        "--store-token-features",
        action="store_true",
        help="Also store per-language-token hidden states and masks for Route B tokenizer diagnostics.",
    )
    parser.add_argument(
        "--language-token-storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Storage dtype for language_token_features when --store-token-features is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _default_output_dir(input_cache_dir: Path, policy_checkpoint_dir: Path) -> Path:
    policy_tag = policy_checkpoint_dir.name.replace("/", "_")
    return input_cache_dir.with_name(f"{input_cache_dir.name}_lang_{policy_tag}")


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest = read_json(cache_dir / "manifest.json")
    if manifest.get("schema_version") != FEATURE_CACHE_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unexpected manifest schema {manifest.get('schema_version')!r}; expected {FEATURE_CACHE_MANIFEST_SCHEMA!r}"
        )
    return manifest


def _load_target_dataset_indices(cache_dir: Path, manifest: dict[str, Any]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for path in shard_paths(cache_dir, manifest):
        payload = load_shard(path, map_location="cpu")
        tensors = payload["tensors"]
        if "dataset_index" not in tensors:
            raise ValueError(f"Shard {path} has no dataset_index tensor")
        chunks.append(tensors["dataset_index"].long().cpu())
    if not chunks:
        raise ValueError(f"No shards found in {cache_dir}")
    return torch.cat(chunks, dim=0)


def _load_policy_model(
    train_config: _config.TrainConfig,
    policy_checkpoint_dir: Path,
    device: torch.device,
) -> PI0Pytorch:
    model_config = train_config.model
    if hasattr(model_config, "dtype") and model_config.dtype != train_config.pytorch_training_precision:
        model_config = dataclasses.replace(model_config, dtype=train_config.pytorch_training_precision)
    model = PI0Pytorch(model_config).to(device=device)
    weight_path = policy_checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Policy checkpoint is missing model.safetensors: {weight_path}")
    safetensors.torch.load_model(model, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(requires_grad=False)
    return model


def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    if att_2d_masks.ndim != 3:
        raise ValueError(f"att_2d_masks must be rank 3, got shape {tuple(att_2d_masks.shape)}")
    return torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype name: {name}")


def _processed_row_keep(processed: _model.Observation, *, include_fallback: bool) -> torch.Tensor:
    image_masks = list(processed.image_masks.values())
    image_padding_masks = list(processed.image_padding_mask.values())
    vision_mask, _vision_metadata = build_vision_token_mask(image_masks, image_padding_masks)
    row_keep = vision_mask.sum(dim=1) > 0
    fallback = getattr(processed, "future_teacher_fallback", None)
    if fallback is not None and not include_fallback:
        fallback_bool = fallback.to(device=row_keep.device).bool().view(row_keep.shape[0])
        row_keep = row_keep & ~fallback_bool
    return row_keep.bool()


@torch.no_grad()
def _language_features_batch(
    model: PI0Pytorch,
    processed: _model.Observation,
    *,
    vla_layer: int,
    store_token_features: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    images = list(processed.images.values())
    img_masks = list(processed.image_masks.values())
    lang_tokens = processed.tokenized_prompt
    lang_masks = processed.tokenized_prompt_mask

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    img_len = int(prefix_embs.shape[1] - lang_tokens.shape[1])
    if img_len < 0:
        raise RuntimeError(f"Computed negative image prefix length {img_len}")
    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    prefix_att_2d_masks_4d = _prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _outputs, _past_key_values, hidden_states = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        output_hidden_states=True,
    )
    prefix_hidden = hidden_states[vla_layer]
    if isinstance(prefix_hidden, tuple):
        prefix_hidden = prefix_hidden[0]
    prefix_hidden = prefix_hidden.float()
    if img_len >= prefix_hidden.shape[1]:
        pooled = torch.zeros(
            prefix_hidden.shape[0], prefix_hidden.shape[-1], dtype=torch.float32, device=prefix_hidden.device
        )
        token_features = None
        token_mask_bool = None
        if store_token_features:
            token_features = pooled[:, None, :]
            token_mask_bool = torch.zeros(
                prefix_hidden.shape[0],
                1,
                dtype=torch.bool,
                device=prefix_hidden.device,
            )
        return pooled, token_features, token_mask_bool
    lang_hidden = prefix_hidden[:, img_len:, :]
    lang_mask_bool = prefix_pad_masks[:, img_len:].to(device=lang_hidden.device).bool()
    lang_mask = lang_mask_bool.to(dtype=lang_hidden.dtype).unsqueeze(-1)
    denom = lang_mask.sum(dim=1).clamp_min(1.0)
    pooled = (lang_hidden * lang_mask).sum(dim=1) / denom
    if not store_token_features:
        return pooled, None, None
    return pooled, lang_hidden.float(), lang_mask_bool


def _stats(value: torch.Tensor) -> dict[str, float | list[int]]:
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
        "abs_mean": float(value.abs().mean()),
    }


def _collect_language_features(
    train_config: _config.TrainConfig,
    manifest: dict[str, Any],
    target_dataset_indices: torch.Tensor,
    policy: PI0Pytorch,
    *,
    device: torch.device,
    seed: int,
    num_workers: int,
    vla_layer: int,
    store_token_features: bool,
    language_token_storage_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    config = dataclasses.replace(
        train_config,
        batch_size=int(manifest["batch_size"]),
        num_workers=int(num_workers),
        seed=int(seed),
    )
    loader = _create_cache_loader(config, shuffle=bool(manifest["shuffle_data"]))
    include_fallback = bool(manifest.get("include_fallback", False))
    num_batches = int(manifest["num_batches_encoded"])

    language_chunks: list[torch.Tensor] = []
    language_token_chunks: list[torch.Tensor] = []
    language_token_mask_chunks: list[torch.Tensor] = []
    cursor = 0
    total = int(target_dataset_indices.numel())
    batch_summaries: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches or cursor >= total:
            break
        LOGGER.info("Extracting language features for replay batch %s/%s", batch_idx + 1, num_batches)
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: as_device_tree(x, device), observation)
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=True, get_wo_aug=False)
        row_keep = _processed_row_keep(processed, include_fallback=include_fallback).to(device=device)
        kept = int(row_keep.sum().detach().cpu())
        batch_dataset_index = _metadata_tensor(batch, "index", device).long()
        kept_dataset_index = batch_dataset_index[row_keep].detach().cpu()
        if kept == 0:
            batch_summaries.append(
                {
                    "batch_idx": int(batch_idx),
                    "kept_samples": 0,
                    "cursor_before": int(cursor),
                    "cursor_after": int(cursor),
                }
            )
            continue

        target_slice = target_dataset_indices[cursor : cursor + kept].cpu()
        if target_slice.shape != kept_dataset_index.shape or not torch.equal(target_slice, kept_dataset_index):
            mismatch = (target_slice != kept_dataset_index).nonzero(as_tuple=False).flatten()
            first = int(mismatch[0]) if int(mismatch.numel()) else 0
            raise RuntimeError(
                "Replay dataloader order does not match source cache at "
                f"batch={batch_idx}, cursor={cursor}, local_mismatch={first}, "
                f"target={int(target_slice[first]) if target_slice.numel() else None}, "
                f"replayed={int(kept_dataset_index[first]) if kept_dataset_index.numel() else None}"
            )

        language_features, language_token_features, language_token_mask = _language_features_batch(
            policy,
            processed,
            vla_layer=vla_layer,
            store_token_features=store_token_features,
        )
        language_chunks.append(language_features[row_keep].detach().cpu().float())
        if store_token_features:
            if language_token_features is None or language_token_mask is None:
                raise RuntimeError("Token feature collection was requested but language token tensors are missing.")
            language_token_chunks.append(
                language_token_features[row_keep].detach().cpu().to(dtype=language_token_storage_dtype)
            )
            language_token_mask_chunks.append(language_token_mask[row_keep].detach().cpu().bool())
        cursor += kept
        batch_summaries.append(
            {
                "batch_idx": int(batch_idx),
                "kept_samples": kept,
                "cursor_before": int(cursor - kept),
                "cursor_after": int(cursor),
                "first_dataset_index": int(kept_dataset_index[0]),
                "last_dataset_index": int(kept_dataset_index[-1]),
            }
        )

    if cursor != total:
        raise RuntimeError(f"Only matched {cursor} samples, but source cache contains {total} samples")
    language_features = torch.cat(language_chunks, dim=0)
    if language_features.shape[:1] != (total,):
        raise RuntimeError(f"Language feature shape mismatch: language={tuple(language_features.shape)}, total={total}")
    language_token_features_out = None
    language_token_mask_out = None
    if store_token_features:
        language_token_features_out = torch.cat(language_token_chunks, dim=0)
        language_token_mask_out = torch.cat(language_token_mask_chunks, dim=0)
        if language_token_features_out.shape[:1] != (total,) or language_token_mask_out.shape[:1] != (total,):
            raise RuntimeError(
                "Language token feature shape mismatch: "
                f"features={tuple(language_token_features_out.shape)}, "
                f"mask={tuple(language_token_mask_out.shape)}, total={total}"
            )
        if language_token_features_out.shape[:2] != language_token_mask_out.shape:
            raise RuntimeError(
                "Language token feature/mask shape mismatch: "
                f"{tuple(language_token_features_out.shape)} vs {tuple(language_token_mask_out.shape)}"
            )
    summary = {
        "num_replay_batches": len(batch_summaries),
        "num_language_samples": total,
        "vla_layer": int(vla_layer),
        "language_features": _stats(language_features),
        "batch_summaries_head": batch_summaries[:5],
        "batch_summaries_tail": batch_summaries[-5:],
    }
    if store_token_features and language_token_features_out is not None and language_token_mask_out is not None:
        summary["language_token_features"] = _stats(language_token_features_out.float())
        summary["language_token_mask"] = {
            "shape": list(language_token_mask_out.shape),
            "valid_tokens": int(language_token_mask_out.sum().item()),
            "valid_ratio": float(language_token_mask_out.float().mean().item()),
        }
        summary["language_token_storage_dtype"] = str(language_token_features_out.dtype).replace("torch.", "")
    return language_features, language_token_features_out, language_token_mask_out, summary


def _write_augmented_cache(
    input_cache_dir: Path,
    output_cache_dir: Path,
    manifest: dict[str, Any],
    language_features: torch.Tensor,
    language_token_features: torch.Tensor | None,
    language_token_mask: torch.Tensor | None,
    *,
    policy_checkpoint_dir: Path,
    vla_layer: int,
    replay_summary: dict[str, Any],
) -> None:
    if output_cache_dir.resolve() == input_cache_dir.resolve():
        raise ValueError("Output cache directory must differ from input cache directory")
    if output_cache_dir.exists():
        raise FileExistsError(f"Output cache already exists: {output_cache_dir}")
    (output_cache_dir / "shards").mkdir(parents=True, exist_ok=False)

    cursor = 0
    new_shards: list[dict[str, Any]] = []
    for path in shard_paths(input_cache_dir, manifest):
        payload = load_shard(path, map_location="cpu")
        tensors = dict(payload["tensors"])
        sample_count = int(tensors["current_tokens"].shape[0])
        language_slice = language_features[cursor : cursor + sample_count].contiguous()
        if language_slice.shape[:1] != (sample_count,):
            raise RuntimeError(f"Bad language slice shape for shard {path}")
        tensors["language_features"] = language_slice
        if language_token_features is not None or language_token_mask is not None:
            if language_token_features is None or language_token_mask is None:
                raise ValueError("language_token_features and language_token_mask must be provided together")
            token_slice = language_token_features[cursor : cursor + sample_count].contiguous()
            mask_slice = language_token_mask[cursor : cursor + sample_count].contiguous()
            if token_slice.shape[:1] != (sample_count,) or mask_slice.shape[:1] != (sample_count,):
                raise RuntimeError(f"Bad language token slice shape for shard {path}")
            if token_slice.shape[:2] != mask_slice.shape:
                raise RuntimeError(
                    f"Bad language token feature/mask slice shape for shard {path}: "
                    f"{tuple(token_slice.shape)} vs {tuple(mask_slice.shape)}"
                )
            tensors["language_token_features"] = token_slice
            tensors["language_token_mask"] = mask_slice
        validate_feature_tensors(
            tensors["current_tokens"],
            tensors["future_tokens"],
            tensors["token_mask"],
            tensors["horizon_ids"],
            tensors["future_teacher_fallback"],
        )

        rel_path = Path("shards") / path.name
        payload["created_at"] = dt.datetime.now(tz=dt.UTC).isoformat()
        payload["source_cache_dir"] = str(input_cache_dir)
        payload["language_feature_policy_checkpoint"] = str(policy_checkpoint_dir)
        payload["language_feature_vla_layer"] = int(vla_layer)
        payload["tensors"] = tensors
        torch.save(payload, output_cache_dir / rel_path)

        old_shard = next(item for item in manifest["shards"] if item["path"] == str(Path("shards") / path.name))
        new_shard = dict(old_shard)
        new_shard["path"] = str(rel_path)
        new_shard["language_features_shape"] = list(language_slice.shape)
        if language_token_features is not None and language_token_mask is not None:
            new_shard["language_token_features_shape"] = list(token_slice.shape)
            new_shard["language_token_mask_shape"] = list(mask_slice.shape)
        new_shards.append(new_shard)
        cursor += sample_count

    if cursor != int(language_features.shape[0]):
        raise RuntimeError(f"Wrote {cursor} samples, expected {int(language_features.shape[0])}")

    new_manifest = copy.deepcopy(manifest)
    metadata_fields = list(new_manifest.get("metadata_fields", []))
    if "language_features" not in metadata_fields:
        metadata_fields.append("language_features")
    if language_token_features is not None and language_token_mask is not None:
        for field in ("language_token_features", "language_token_mask"):
            if field not in metadata_fields:
                metadata_fields.append(field)
    new_manifest.update(
        {
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "source_cache_dir": str(input_cache_dir),
            "language_feature_policy_checkpoint": str(policy_checkpoint_dir),
            "language_feature_vla_layer": int(vla_layer),
            "metadata_fields": metadata_fields,
            "shards": new_shards,
            "language_feature_summary": replay_summary,
            "stores_language_token_features": language_token_features is not None,
        }
    )
    write_json(output_cache_dir / "manifest.json", new_manifest)


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    input_cache_dir = Path(args.input_cache_dir)
    policy_checkpoint_dir = Path(args.policy_checkpoint_dir)
    output_cache_dir = (
        Path(args.output_cache_dir)
        if args.output_cache_dir is not None
        else _default_output_dir(input_cache_dir, policy_checkpoint_dir)
    )
    if output_cache_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_cache_dir)

    manifest = _load_manifest(input_cache_dir)
    config_name = str(args.config_name or manifest["config_name"])
    seed = int(args.seed if args.seed is not None else manifest["seed"])
    device = select_device(str(args.device))
    target_dataset_indices = _load_target_dataset_indices(input_cache_dir, manifest)
    train_config = _config.get_config(config_name)
    manifest_offset = manifest.get("future_teacher_obs_offset")
    if manifest_offset is not None:
        if not dataclasses.is_dataclass(train_config.data):
            raise TypeError("Config data object must be a dataclass to replay future_teacher_obs_offset.")
        if not hasattr(train_config.data, "future_teacher_obs_offset"):
            raise ValueError(
                "Input cache manifest has future_teacher_obs_offset, but this config data object has no "
                "future_teacher_obs_offset field."
            )
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, future_teacher_obs_offset=int(manifest_offset)),
        )
    vla_layer = int(args.vla_layer if args.vla_layer is not None else (train_config.vla_layers_align or 12))
    policy = _load_policy_model(train_config, policy_checkpoint_dir, device)

    language_features, language_token_features, language_token_mask, replay_summary = _collect_language_features(
        train_config,
        manifest,
        target_dataset_indices,
        policy,
        device=device,
        seed=seed,
        num_workers=int(args.num_workers),
        vla_layer=vla_layer,
        store_token_features=bool(args.store_token_features),
        language_token_storage_dtype=_dtype_from_name(str(args.language_token_storage_dtype)),
    )
    _write_augmented_cache(
        input_cache_dir,
        output_cache_dir,
        manifest,
        language_features,
        language_token_features,
        language_token_mask,
        policy_checkpoint_dir=policy_checkpoint_dir,
        vla_layer=vla_layer,
        replay_summary=replay_summary,
    )
    summary = {
        "input_cache_dir": str(input_cache_dir),
        "output_cache_dir": str(output_cache_dir),
        "config_name": config_name,
        "policy_checkpoint_dir": str(policy_checkpoint_dir),
        "device": str(device),
        "seed": seed,
        **replay_summary,
    }
    write_json(output_cache_dir / "language_feature_build_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote language-feature cache to %s", output_cache_dir)


if __name__ == "__main__":
    main()
