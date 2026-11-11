"""Inspect real LIBERO-plus VGGT teacher tokens for scheme3 delta tokenizer work.

This script is a small diagnostic, not a training entrypoint.  It reuses the
existing pi0.5 LIBERO-plus dataloader and VGGT preprocessing/pooling path, then
prints and optionally saves a JSON summary of current/future teacher token
shapes and masks.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import jax
import torch
import torch.nn.functional as F  # noqa: N812

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.training.config as _config
import openpi.training.data_loader as _data
from vggt.heads.utils import custom_pooling
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images_from_openpi

LOGGER = logging.getLogger("check_delta_vggt_inputs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_libero_plus_geo")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--skip-vggt", action="store_true", help="Only inspect dataloader/preprocessed image shapes.")
    parser.add_argument(
        "--no-load-vggt-weights",
        action="store_true",
        help="Instantiate VGGT but do not load the configured checkpoint; useful only for shape debugging.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults to artifacts/scheme3_vggt_input_checks/<timestamp>.json.",
    )
    return parser.parse_args()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def select_device(device_arg: str) -> torch.device:
    return _select_device(device_arg)


def _as_device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def as_device_tree(value: Any, device: torch.device) -> Any:
    return _as_device_tree(value, device)


def _shape(value: torch.Tensor | None) -> list[int] | None:
    if value is None:
        return None
    return list(value.shape)


def _tensor_stats(value: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    value = value.detach().float()
    if mask is not None:
        if mask.shape != value.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match token tensor {tuple(value.shape)}")
        value = value[mask.bool()]
    if value.numel() == 0:
        return {"numel": 0}
    return {
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
        "abs_mean": float(value.abs().mean().cpu()),
    }


def _token_norm_stats(value: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    token_norm = value.detach().float().norm(dim=-1)
    if mask is not None:
        token_norm = token_norm[mask.bool()]
    if token_norm.numel() == 0:
        return {"numel": 0}
    return {
        "shape": list(token_norm.shape),
        "mean": float(token_norm.mean().cpu()),
        "std": float(token_norm.std(unbiased=False).cpu()),
        "min": float(token_norm.min().cpu()),
        "max": float(token_norm.max().cpu()),
    }


def _build_vision_token_mask(
    image_masks: list[torch.Tensor],
    image_padding_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(image_masks) != len(image_padding_masks):
        raise ValueError(f"image mask count {len(image_masks)} != padding mask count {len(image_padding_masks)}")
    if not image_masks:
        raise ValueError("At least one image mask is required")

    image_mask_stack = torch.stack(image_masks, dim=1).bool()
    padding_mask_stack = torch.stack(image_padding_masks, dim=1).bool()
    if padding_mask_stack.ndim != 4:
        raise ValueError(f"padding masks must stack to [B, C, H, W], got {tuple(padding_mask_stack.shape)}")

    patch_h = padding_mask_stack.shape[-2] // 14
    patch_w = padding_mask_stack.shape[-1] // 14
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Invalid image padding mask shape {tuple(padding_mask_stack.shape)} for patch size 14")

    tokens_per_image = patch_h * patch_w
    align_mask = torch.repeat_interleave(image_mask_stack, repeats=tokens_per_image, dim=1)
    padding_downsampled = F.interpolate(
        padding_mask_stack.float(),
        size=(patch_h, patch_w),
        mode="nearest",
    ).bool()
    padding_token_mask = padding_downsampled.flatten(start_dim=1)
    if align_mask.shape != padding_token_mask.shape:
        raise RuntimeError(
            f"vision mask shape mismatch: align_mask={tuple(align_mask.shape)}, "
            f"padding_token_mask={tuple(padding_token_mask.shape)}"
        )
    vision_mask = align_mask & padding_token_mask
    metadata = {
        "num_images": len(image_masks),
        "patch_hw_from_openpi_padding": [int(patch_h), int(patch_w)],
        "tokens_per_image": int(tokens_per_image),
        "vision_tokens": int(vision_mask.shape[1]),
        "valid_token_ratio": float(vision_mask.float().mean().cpu()),
        "valid_tokens_per_sample": [int(x) for x in vision_mask.sum(dim=1).cpu().tolist()],
    }
    return vision_mask, metadata


def build_vision_token_mask(
    image_masks: list[torch.Tensor],
    image_padding_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    return _build_vision_token_mask(image_masks, image_padding_masks)


def _load_vggt(config: _config.TrainConfig, device: torch.device, *, load_weights: bool) -> VGGT:
    model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    ).to(device)
    model.eval()
    if load_weights:
        if config.vggt_weight_path is None:
            raise ValueError("config.vggt_weight_path is None")
        vggt_path = Path(config.vggt_weight_path) / "model.pt"
        if not vggt_path.exists():
            raise FileNotFoundError(f"VGGT weight file not found: {vggt_path}")
        state = torch.load(vggt_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_vggt(config: _config.TrainConfig, device: torch.device, *, load_weights: bool) -> VGGT:
    return _load_vggt(config, device, load_weights=load_weights)


def _encode_vggt_hidden(
    images: list[torch.Tensor],
    vggt: VGGT,
    *,
    vision_token_count: int,
    pooling_func: str,
    use_vggt_pe: bool,
    vggt_layers_align: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    resized = preprocess_images_from_openpi(images)
    autocast_ctx = (
        torch.autocast(resized.device.type, dtype=torch.bfloat16)
        if resized.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_ctx, torch.no_grad():
        output = vggt(resized)

    features = output["features"][vggt_layers_align]
    patch_start_idx = output["patch_start_idx"]
    original_img = output["images"]
    hidden = features[:, :, patch_start_idx:, :]
    height, width = original_img.shape[-2:]
    patch_h = height // vggt.patch_size
    patch_w = width // vggt.patch_size
    reference = torch.empty(hidden.shape[0], vision_token_count, 1, dtype=torch.float32, device=hidden.device)
    pooled = custom_pooling(
        hidden,
        (patch_h, patch_w),
        (height, width),
        reference,
        pooling_func,
        use_vggt_pe,
    )
    metadata = {
        "vggt_resized_images": list(resized.shape),
        "vggt_feature_shape_before_pooling": list(features.shape),
        "vggt_patch_start_idx": int(patch_start_idx),
        "vggt_hidden_patch_shape": list(hidden.shape),
        "vggt_original_img_hw": [int(height), int(width)],
        "vggt_patch_hw": [int(patch_h), int(patch_w)],
    }
    return pooled.detach(), metadata


def encode_vggt_hidden(
    images: list[torch.Tensor],
    vggt: VGGT,
    *,
    vision_token_count: int,
    pooling_func: str,
    use_vggt_pe: bool,
    vggt_layers_align: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    return _encode_vggt_hidden(
        images,
        vggt,
        vision_token_count=vision_token_count,
        pooling_func=pooling_func,
        use_vggt_pe=use_vggt_pe,
        vggt_layers_align=vggt_layers_align,
    )


def _summarize_batch(
    batch_idx: int,
    observation,
    actions: torch.Tensor,
    config: _config.TrainConfig,
    device: torch.device,
    vggt: VGGT | None,
) -> dict[str, Any]:
    observation = jax.tree.map(lambda x: _as_device_tree(x, device), observation)
    actions = actions.to(device)
    processed = _preprocessing.preprocess_observation_pytorch(observation, train=True, get_wo_aug=False)

    teacher_images = list(processed.teacher_images.values()) if hasattr(processed, "teacher_images") else None
    future_teacher_images = (
        list(processed.future_teacher_images.values()) if hasattr(processed, "future_teacher_images") else None
    )
    if teacher_images is None or future_teacher_images is None:
        raise ValueError("Processed observation does not contain teacher/future_teacher images")

    image_masks = list(processed.image_masks.values())
    image_padding_masks = list(processed.image_padding_mask.values())
    vision_mask, vision_metadata = _build_vision_token_mask(image_masks, image_padding_masks)

    fallback = getattr(processed, "future_teacher_fallback", None)
    fallback_bool = None if fallback is None else fallback.to(device=device).bool().view(actions.shape[0])

    summary: dict[str, Any] = {
        "batch_idx": batch_idx,
        "actions_shape": list(actions.shape),
        "state_shape": _shape(processed.state),
        "image_keys": list(processed.images.keys()),
        "input_image_shapes": {key: list(value.shape) for key, value in processed.images.items()},
        "teacher_image_shapes": {key: list(value.shape) for key, value in processed.teacher_images.items()},
        "future_teacher_image_shapes": {
            key: list(value.shape) for key, value in processed.future_teacher_images.items()
        },
        "image_mask_shapes": {key: list(value.shape) for key, value in processed.image_masks.items()},
        "image_padding_mask_shapes": {key: list(value.shape) for key, value in processed.image_padding_mask.items()},
        "future_teacher_fallback_shape": _shape(fallback),
        "future_teacher_fallback_values": None
        if fallback_bool is None
        else [bool(x) for x in fallback_bool.cpu().tolist()],
        "future_teacher_fallback_count": None if fallback_bool is None else int(fallback_bool.sum().cpu()),
        "vision_mask": vision_metadata,
    }

    if vggt is None:
        return summary

    pooling_func = "bilinear" if config.pooling_func is None else config.pooling_func
    use_vggt_pe = True if config.use_vggt_pe is None else bool(config.use_vggt_pe)
    vggt_layers_align = -1 if config.vggt_layers_align is None else int(config.vggt_layers_align)
    current_hidden, current_meta = _encode_vggt_hidden(
        teacher_images,
        vggt,
        vision_token_count=int(vision_mask.shape[1]),
        pooling_func=pooling_func,
        use_vggt_pe=use_vggt_pe,
        vggt_layers_align=vggt_layers_align,
    )
    future_hidden, future_meta = _encode_vggt_hidden(
        future_teacher_images,
        vggt,
        vision_token_count=int(vision_mask.shape[1]),
        pooling_func=pooling_func,
        use_vggt_pe=use_vggt_pe,
        vggt_layers_align=vggt_layers_align,
    )
    if current_hidden.shape != future_hidden.shape:
        raise RuntimeError(
            f"current/future hidden shape mismatch: {tuple(current_hidden.shape)} vs {tuple(future_hidden.shape)}"
        )
    if current_hidden.shape[:2] != vision_mask.shape:
        raise RuntimeError(
            f"VGGT token shape {tuple(current_hidden.shape[:2])} does not match vision_mask {tuple(vision_mask.shape)}"
        )
    expected_dim = int(config.geo_teacher_dim or config.vggt_dim)
    if current_hidden.shape[-1] != expected_dim:
        raise RuntimeError(f"VGGT dim {current_hidden.shape[-1]} does not match expected {expected_dim}")

    valid_batch_mask = None if fallback_bool is None else ~fallback_bool
    token_mask_for_delta = vision_mask if valid_batch_mask is None else vision_mask & valid_batch_mask[:, None]
    delta = future_hidden - current_hidden
    summary.update(
        {
            "vggt_current_meta": current_meta,
            "vggt_future_meta": future_meta,
            "current_hidden_shape": list(current_hidden.shape),
            "future_hidden_shape": list(future_hidden.shape),
            "current_hidden_dtype": str(current_hidden.dtype),
            "future_hidden_dtype": str(future_hidden.dtype),
            "current_hidden_stats": _tensor_stats(current_hidden, vision_mask),
            "future_hidden_stats": _tensor_stats(future_hidden, vision_mask),
            "current_token_norm": _token_norm_stats(current_hidden, vision_mask),
            "future_token_norm": _token_norm_stats(future_hidden, vision_mask),
            "delta_valid_token_stats": _tensor_stats(delta, token_mask_for_delta),
            "delta_valid_token_norm": _token_norm_stats(delta, token_mask_for_delta),
        }
    )
    return summary


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    device = _select_device(args.device)
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=args.num_workers)

    LOGGER.info(
        "Using config=%s batch_size=%s num_batches=%s device=%s", config.name, args.batch_size, args.num_batches, device
    )
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=args.num_batches,
    )
    vggt = None
    if not args.skip_vggt:
        vggt = _load_vggt(config, device, load_weights=not args.no_load_vggt_weights)
        LOGGER.info("Loaded VGGT on %s (weights=%s)", device, not args.no_load_vggt_weights)

    batches = []
    for batch_idx, (observation, actions) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        LOGGER.info("Inspecting batch %s", batch_idx)
        batches.append(_summarize_batch(batch_idx, observation, actions, config, device, vggt))

    output = {
        "schema_version": "scheme3_vggt_input_check_v1",
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "config_name": config.name,
        "repo_id": config.data.repo_id,
        "future_teacher_obs_offset": config.data.future_teacher_obs_offset,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "device": str(device),
        "vggt_loaded": vggt is not None,
        "vggt_weights_loaded": vggt is not None and not args.no_load_vggt_weights,
        "batches": batches,
    }
    if args.output_json is None:
        output_dir = Path("artifacts/scheme3_vggt_input_checks")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{config.name}_b{args.batch_size}_n{args.num_batches}_{timestamp}.json"
    else:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
