"""Augment a scheme3 VGGT feature cache with pi0.5 estimated action endpoints.

This script replays the data order recorded by a feature cache manifest, runs a
frozen pi0.5 policy on the matching normalized observations/actions, computes

    action_hat = x_t - t * v_hat

and writes a new cache containing the original tensors plus:

- estimated_actions: [S, H_action, A_dim]
- estimated_action_times: [S]

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
from openpi.models_pytorch.flow_matching_utils import estimate_action_endpoint
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.training.config as _config

LOGGER = logging.getLogger("add_estimated_actions_to_delta_cache")

DEFAULT_INPUT_CACHE = "artifacts/delta_vggt_cache/s3_m4_n2048_seed31_gpu0_cache_b2_target2048_shuffle_seed31"
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
    parser.add_argument(
        "--estimation-seed",
        type=int,
        default=None,
        help="Seed for flow noise/time sampling. Defaults to seed + 1701.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _default_output_dir(input_cache_dir: Path, policy_checkpoint_dir: Path) -> Path:
    policy_tag = policy_checkpoint_dir.name.replace("/", "_")
    return input_cache_dir.with_name(f"{input_cache_dir.name}_estact_{policy_tag}")


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
def _estimate_batch_endpoints(
    model: PI0Pytorch,
    processed: _model.Observation,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H_action, A_dim], got {tuple(actions.shape)}")

    images = list(processed.images.values())
    img_masks = list(processed.image_masks.values())
    lang_tokens = processed.tokenized_prompt
    lang_masks = processed.tokenized_prompt_mask
    state = processed.state
    device = actions.device
    batch_size = int(actions.shape[0])

    noise = model.sample_noise(actions.shape, device)
    time = model.sample_time(batch_size, device)
    if time.shape != (batch_size,):
        raise RuntimeError(f"sample_time returned shape {tuple(time.shape)}, expected {(batch_size,)}")
    time_expanded = time.view(batch_size, 1, 1).expand_as(actions)
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = _prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    (outputs_prefix, suffix_out), _ = model.paligemma_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    del outputs_prefix
    suffix_out = suffix_out[:, -model.config.action_horizon :].to(dtype=torch.float32)
    v_hat = model.action_out_proj(suffix_out)
    if v_hat.shape != actions.shape:
        raise RuntimeError(f"v_hat shape {tuple(v_hat.shape)} does not match actions shape {tuple(actions.shape)}")
    action_hat = estimate_action_endpoint(x_t, time, v_hat, detach=True)
    return action_hat, time.detach()


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


def _collect_estimated_actions(
    train_config: _config.TrainConfig,
    manifest: dict[str, Any],
    target_dataset_indices: torch.Tensor,
    policy: PI0Pytorch,
    *,
    device: torch.device,
    seed: int,
    estimation_seed: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    config = dataclasses.replace(
        train_config,
        batch_size=int(manifest["batch_size"]),
        num_workers=int(num_workers),
        seed=int(seed),
    )
    loader = _create_cache_loader(config, shuffle=bool(manifest["shuffle_data"]))
    include_fallback = bool(manifest.get("include_fallback", False))
    num_batches = int(manifest["num_batches_encoded"])
    _set_seed(estimation_seed, device)

    estimated_chunks: list[torch.Tensor] = []
    time_chunks: list[torch.Tensor] = []
    cursor = 0
    total = int(target_dataset_indices.numel())
    batch_summaries: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches or cursor >= total:
            break
        LOGGER.info("Estimating actions for replay batch %s/%s", batch_idx + 1, num_batches)
        observation = _model.Observation.from_dict(batch)
        observation = jax.tree.map(lambda x: as_device_tree(x, device), observation)
        actions = batch["actions"].to(device=device).float()
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

        action_hat, time = _estimate_batch_endpoints(policy, processed, actions)
        estimated_chunks.append(action_hat[row_keep].detach().cpu().float())
        time_chunks.append(time[row_keep].detach().cpu().float())
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
    estimated_actions = torch.cat(estimated_chunks, dim=0)
    estimated_times = torch.cat(time_chunks, dim=0)
    if estimated_actions.shape[:1] != (total,) or estimated_times.shape != (total,):
        raise RuntimeError(
            "Estimated tensor shape mismatch: "
            f"actions={tuple(estimated_actions.shape)}, times={tuple(estimated_times.shape)}, total={total}"
        )
    summary = {
        "num_replay_batches": len(batch_summaries),
        "num_estimated_samples": total,
        "estimated_actions": _stats(estimated_actions),
        "estimated_action_times": _stats(estimated_times),
        "batch_summaries_head": batch_summaries[:5],
        "batch_summaries_tail": batch_summaries[-5:],
    }
    return estimated_actions, estimated_times, summary


def _write_augmented_cache(
    input_cache_dir: Path,
    output_cache_dir: Path,
    manifest: dict[str, Any],
    estimated_actions: torch.Tensor,
    estimated_times: torch.Tensor,
    *,
    policy_checkpoint_dir: Path,
    estimation_seed: int,
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
        action_slice = estimated_actions[cursor : cursor + sample_count].contiguous()
        time_slice = estimated_times[cursor : cursor + sample_count].contiguous()
        if action_slice.shape[:1] != (sample_count,) or time_slice.shape != (sample_count,):
            raise RuntimeError(f"Bad estimated slice shape for shard {path}")
        tensors["estimated_actions"] = action_slice
        tensors["estimated_action_times"] = time_slice
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
        payload["estimated_action_policy_checkpoint"] = str(policy_checkpoint_dir)
        payload["tensors"] = tensors
        torch.save(payload, output_cache_dir / rel_path)

        old_shard = next(item for item in manifest["shards"] if item["path"] == str(Path("shards") / path.name))
        new_shard = dict(old_shard)
        new_shard["path"] = str(rel_path)
        new_shard["estimated_actions_shape"] = list(action_slice.shape)
        new_shard["estimated_action_times_shape"] = list(time_slice.shape)
        new_shards.append(new_shard)
        cursor += sample_count

    if cursor != int(estimated_actions.shape[0]):
        raise RuntimeError(f"Wrote {cursor} samples, expected {int(estimated_actions.shape[0])}")

    new_manifest = copy.deepcopy(manifest)
    metadata_fields = list(new_manifest.get("metadata_fields", []))
    for field in ("estimated_actions", "estimated_action_times"):
        if field not in metadata_fields:
            metadata_fields.append(field)
    new_manifest.update(
        {
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "source_cache_dir": str(input_cache_dir),
            "estimated_action_policy_checkpoint": str(policy_checkpoint_dir),
            "estimated_action_formula": "action_hat = x_t - t * v_hat",
            "estimated_action_stop_gradient": True,
            "estimated_action_seed": int(estimation_seed),
            "metadata_fields": metadata_fields,
            "shards": new_shards,
            "estimated_action_summary": replay_summary,
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
    estimation_seed = int(args.estimation_seed if args.estimation_seed is not None else seed + 1701)
    device = select_device(str(args.device))
    target_dataset_indices = _load_target_dataset_indices(input_cache_dir, manifest)
    train_config = _config.get_config(config_name)
    policy = _load_policy_model(train_config, policy_checkpoint_dir, device)

    estimated_actions, estimated_times, replay_summary = _collect_estimated_actions(
        train_config,
        manifest,
        target_dataset_indices,
        policy,
        device=device,
        seed=seed,
        estimation_seed=estimation_seed,
        num_workers=int(args.num_workers),
    )
    _write_augmented_cache(
        input_cache_dir,
        output_cache_dir,
        manifest,
        estimated_actions,
        estimated_times,
        policy_checkpoint_dir=policy_checkpoint_dir,
        estimation_seed=estimation_seed,
        replay_summary=replay_summary,
    )
    summary = {
        "input_cache_dir": str(input_cache_dir),
        "output_cache_dir": str(output_cache_dir),
        "config_name": config_name,
        "policy_checkpoint_dir": str(policy_checkpoint_dir),
        "device": str(device),
        "seed": seed,
        "estimation_seed": estimation_seed,
        **replay_summary,
    }
    write_json(output_cache_dir / "estimated_action_build_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote estimated-action cache to %s", output_cache_dir)


if __name__ == "__main__":
    main()
