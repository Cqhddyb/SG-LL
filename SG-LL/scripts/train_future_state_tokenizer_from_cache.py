"""Train the matched future-state tokenizer for the semantics frame baseline.

The script deliberately replays the saved split and optimization protocol from
an existing transition-tokenizer checkpoint.  It trains only
``FutureStateTokenizer`` on ``S_t+h`` and never passes sample-specific
``S_t`` into its encoder or decoder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

from delta_vggt_cache_utils import FeatureCacheTensors
from delta_vggt_cache_utils import feature_cache_shape_summary
from delta_vggt_cache_utils import load_feature_cache
from delta_vggt_cache_utils import write_json
import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import geometry_distance
from openpi.models_pytorch.future_state_tokenizer import FutureStateTokenizer
from openpi.models_pytorch.future_state_tokenizer import future_state_tokenizer_loss

LOGGER = logging.getLogger("train_future_state_tokenizer_from_cache")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reference-tokenizer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--overwrite", action="store_true")

    # ``None`` means "take the exact value saved in the reference checkpoint".
    # A supplied value is rejected when it would change the matched protocol.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--smooth-l1-weight", type=float, default=None)
    parser.add_argument("--cosine-weight", type=float, default=None)
    parser.add_argument("--relation-weight", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_reference_checkpoint(path: Path) -> tuple[DeltaTokenizerConfig, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload or "summary" not in payload:
        raise ValueError(f"reference tokenizer checkpoint is missing config or summary: {path}")
    if not isinstance(payload["config"], dict) or not isinstance(payload["summary"], dict):
        raise ValueError(f"reference tokenizer checkpoint has malformed config or summary: {path}")
    summary = payload["summary"]
    reference_args = summary.get("args")
    if not isinstance(reference_args, dict):
        raise ValueError(f"reference tokenizer summary has no args dictionary: {path}")
    return DeltaTokenizerConfig(**payload["config"]), summary, reference_args


def _matched_value(
    name: str,
    supplied_value: int | float | None,
    reference_args: dict[str, Any],
) -> int | float:
    if name not in reference_args:
        raise ValueError(f"reference tokenizer args are missing {name!r}")
    reference_value = reference_args[name]
    if supplied_value is None:
        return reference_value
    if isinstance(reference_value, float):
        if float(supplied_value) != float(reference_value):
            raise ValueError(
                f"{name}={supplied_value} changes the reference tokenizer protocol ({reference_value}); omit it instead"
            )
    elif int(supplied_value) != int(reference_value):
        raise ValueError(
            f"{name}={supplied_value} changes the reference tokenizer protocol ({reference_value}); omit it instead"
        )
    return reference_value


def _require_reference_protocol(args: argparse.Namespace, reference_args: dict[str, Any]) -> dict[str, int | float]:
    matched = {
        name: _matched_value(name, getattr(args, name.replace("-", "_")), reference_args)
        for name in (
            "seed",
            "batch_size",
            "train_steps",
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "smooth_l1_weight",
            "cosine_weight",
            "relation_weight",
        )
    }
    if float(reference_args.get("identity_weight", 0.0)) != 0.0:
        raise ValueError(
            "strict frame-token baseline requires a reference tokenizer trained with identity_weight=0.0, "
            f"got {reference_args.get('identity_weight')!r}"
        )
    if bool(reference_args.get("no_zero_init_decoder", False)):
        raise ValueError("strict frame-token baseline requires reference zero_init_decoder=True")
    if int(matched["batch_size"]) <= 0 or int(matched["train_steps"]) <= 0:
        raise ValueError("reference batch_size and train_steps must be positive")
    return matched


def _validate_reference_cache(cache_dir: Path, summary: dict[str, Any]) -> None:
    reference_cache_dir = summary.get("cache_dir")
    if not isinstance(reference_cache_dir, str):
        raise ValueError("reference tokenizer summary has no cache_dir")
    if cache_dir.resolve() != Path(reference_cache_dir).resolve():
        raise ValueError(
            "frame-token baseline must use the exact cache used by the reference transition tokenizer: "
            f"expected {Path(reference_cache_dir).resolve()}, got {cache_dir.resolve()}"
        )


def _reference_split(
    summary: dict[str, Any],
    *,
    sample_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_values = summary.get("train_indices")
    val_values = summary.get("val_indices")
    if not isinstance(train_values, list) or not isinstance(val_values, list):
        raise ValueError("reference tokenizer summary must contain list-valued train_indices and val_indices")
    if not train_values or not val_values:
        raise ValueError("reference tokenizer train/val splits must both be non-empty")
    train_indices = torch.tensor([int(value) for value in train_values], dtype=torch.long, device=device)
    val_indices = torch.tensor([int(value) for value in val_values], dtype=torch.long, device=device)
    all_indices = torch.cat([train_indices, val_indices], dim=0)
    if int(all_indices.min()) < 0 or int(all_indices.max()) >= sample_count:
        raise ValueError(
            f"reference split indices must lie in [0, {sample_count}), got min={int(all_indices.min())}, "
            f"max={int(all_indices.max())}"
        )
    if int(torch.unique(all_indices).numel()) != int(all_indices.numel()):
        raise ValueError("reference tokenizer train/val split has duplicate or overlapping indices")
    return train_indices, val_indices


def _subset(features: FeatureCacheTensors, indices: torch.Tensor) -> FeatureCacheTensors:
    indices = indices.to(device=features.future_tokens.device, dtype=torch.long)
    metadata = {
        key: value.index_select(0, indices)
        for key, value in features.metadata.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (features.future_tokens.shape[0],)
    }
    return FeatureCacheTensors(
        current_tokens=features.current_tokens.index_select(0, indices),
        future_tokens=features.future_tokens.index_select(0, indices),
        token_mask=features.token_mask.index_select(0, indices),
        horizon_ids=features.horizon_ids.index_select(0, indices),
        future_teacher_fallback=features.future_teacher_fallback.index_select(0, indices),
        metadata=metadata,
        manifest=features.manifest,
    )


def _metric_scalars(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _loss_kwargs(protocol: dict[str, int | float]) -> dict[str, float]:
    return {
        "smooth_l1_weight": float(protocol["smooth_l1_weight"]),
        "cosine_weight": float(protocol["cosine_weight"]),
        "relation_weight": float(protocol["relation_weight"]),
    }


def _evaluate(
    tokenizer: FutureStateTokenizer,
    features: FeatureCacheTensors,
    protocol: dict[str, int | float],
    *,
    batch_size: int,
) -> dict[str, float]:
    tokenizer.eval()
    total = int(features.future_tokens.shape[0])
    weighted: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = _subset(features, torch.arange(start, stop, device=features.future_tokens.device))
            result = future_state_tokenizer_loss(
                tokenizer,
                batch.future_tokens,
                token_mask=batch.token_mask,
                horizon_ids=batch.horizon_ids,
                **_loss_kwargs(protocol),
            )
            count = stop - start
            for key, value in _metric_scalars(result.metrics).items():
                weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / total for key, value in weighted.items()}


def _zero_as_future_baseline(
    features: FeatureCacheTensors,
    protocol: dict[str, int | float],
    *,
    relation_max_tokens: int,
    batch_size: int,
) -> dict[str, float]:
    total = int(features.future_tokens.shape[0])
    weighted: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = _subset(features, torch.arange(start, stop, device=features.future_tokens.device))
            zero_future = torch.zeros_like(batch.future_tokens)
            loss, metrics = geometry_distance(
                zero_future,
                batch.future_tokens,
                batch.token_mask,
                relation_max_tokens=relation_max_tokens,
                **_loss_kwargs(protocol),
            )
            scalars = _metric_scalars(metrics)
            scalars["zero_as_future_loss"] = float(loss.detach().cpu())
            count = stop - start
            for key, value in scalars.items():
                weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / total for key, value in weighted.items()}


def _train(
    tokenizer: FutureStateTokenizer,
    train_features: FeatureCacheTensors,
    val_features: FeatureCacheTensors,
    protocol: dict[str, int | float],
    *,
    log_every: int,
    output_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=float(protocol["lr"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(protocol["seed"]))
    train_count = int(train_features.future_tokens.shape[0])
    batch_size = min(int(protocol["batch_size"]), train_count)
    metrics_path = output_dir / "metrics.jsonl"
    rows: list[dict[str, float]] = []

    tokenizer.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(int(protocol["train_steps"])):
            index = torch.randint(train_count, (batch_size,), generator=generator)
            batch = _subset(train_features, index.to(device=train_features.future_tokens.device))
            optimizer.zero_grad(set_to_none=True)
            result = future_state_tokenizer_loss(
                tokenizer,
                batch.future_tokens,
                token_mask=batch.token_mask,
                horizon_ids=batch.horizon_ids,
                **_loss_kwargs(protocol),
            )
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"non-finite frame-tokenizer loss at step {step + 1}: {result.loss.detach()}")
            result.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                tokenizer.parameters(),
                max_norm=float(protocol["grad_clip_norm"]),
            )
            optimizer.step()
            row = {
                "step": float(step + 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()),
                **_metric_scalars(result.metrics),
            }
            rows.append(row)
            metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if step == 0 or (step + 1) % int(log_every) == 0:
                LOGGER.info(
                    "step=%d frame_train_loss=%.6f frame_rec=%.6f",
                    step + 1,
                    row["future_state_tokenizer_loss"],
                    row["future_state_rec_loss"],
                )

    final_train = _evaluate(tokenizer, train_features, protocol, batch_size=batch_size)
    final_val = _evaluate(
        tokenizer,
        val_features,
        protocol,
        batch_size=max(1, min(int(protocol["batch_size"]), int(val_features.future_tokens.shape[0]))),
    )
    return rows, final_train, final_val


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    reference_checkpoint = Path(args.reference_tokenizer_checkpoint)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config, reference_summary, reference_args = _load_reference_checkpoint(reference_checkpoint)
    _validate_reference_cache(cache_dir, reference_summary)
    protocol = _require_reference_protocol(args, reference_args)
    device = _select_device(str(args.device))
    _set_seed(int(protocol["seed"]), device)
    if device.type == "cuda":
        # The explicit device form can fail before CUDA has initialized the
        # current logical device (for example under CUDA_VISIBLE_DEVICES=1).
        torch.cuda.reset_peak_memory_stats()

    write_json(output_dir / "args.json", vars(args))
    features = load_feature_cache(cache_dir, device=device, drop_fallback=True)
    train_indices, val_indices = _reference_split(
        reference_summary,
        sample_count=int(features.future_tokens.shape[0]),
        device=device,
    )
    train_features = _subset(features, train_indices)
    val_features = _subset(features, val_indices)
    tokenizer = FutureStateTokenizer(reference_config).to(device)
    write_json(output_dir / "tokenizer_config.json", tokenizer.config_dict())

    train_batch_size = min(int(protocol["batch_size"]), int(train_features.future_tokens.shape[0]))
    val_batch_size = max(1, min(int(protocol["batch_size"]), int(val_features.future_tokens.shape[0])))
    initial_train = _evaluate(tokenizer, train_features, protocol, batch_size=train_batch_size)
    initial_val = _evaluate(tokenizer, val_features, protocol, batch_size=val_batch_size)
    zero_train = _zero_as_future_baseline(
        train_features,
        protocol,
        relation_max_tokens=reference_config.relation_max_tokens,
        batch_size=train_batch_size,
    )
    zero_val = _zero_as_future_baseline(
        val_features,
        protocol,
        relation_max_tokens=reference_config.relation_max_tokens,
        batch_size=val_batch_size,
    )
    LOGGER.info(
        "initial frame train loss=%.6f val loss=%.6f zero-val=%.6f",
        initial_train["future_state_tokenizer_loss"],
        initial_val["future_state_tokenizer_loss"],
        zero_val["zero_as_future_loss"],
    )
    rows, final_train, final_val = _train(
        tokenizer,
        train_features,
        val_features,
        protocol,
        log_every=int(args.log_every),
        output_dir=output_dir,
    )

    summary: dict[str, Any] = {
        "schema_version": "scheme4_frame_tokenizer_cached_train_v1",
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "method": {
            "formula": "Z_f = E_f(S_t+h); S_hat_t+h = D_f(Z_f)",
            "current_input_policy": "No sample-specific current teacher tokens enter encode or decode.",
            "architecture_parity": "DeltaEncoder/DeltaDecoder with fixed all-zero current slots only.",
        },
        "cache_dir": str(cache_dir),
        "reference_transition_tokenizer_checkpoint": str(reference_checkpoint),
        "reference_transition_tokenizer_cache_dir": reference_summary.get("cache_dir"),
        "reference_transition_tokenizer_args": reference_args,
        "matched_training_protocol": protocol,
        "device": str(device),
        "cache_manifest": {
            key: features.manifest.get(key)
            for key in (
                "schema_version",
                "config_name",
                "repo_id",
                "future_teacher_obs_offset",
                "horizon_id",
                "storage_dtype",
                "total_samples",
            )
        },
        "feature_shapes": feature_cache_shape_summary(features),
        "train_indices": [int(value) for value in train_indices.detach().cpu().tolist()],
        "val_indices": [int(value) for value in val_indices.detach().cpu().tolist()],
        "tokenizer_config": tokenizer.config_dict(),
        "initial_train_metrics": initial_train,
        "initial_val_metrics": initial_val,
        "zero_as_future_train": zero_train,
        "zero_as_future_val": zero_val,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "train_loss_decreased": final_train["future_state_tokenizer_loss"]
        < initial_train["future_state_tokenizer_loss"],
        "val_loss_decreased": final_val["future_state_tokenizer_loss"] < initial_val["future_state_tokenizer_loss"],
        "last_train_row": rows[-1] if rows else None,
        "cuda_max_memory_allocated_bytes": None
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "schema_version": "scheme4_frame_tokenizer_cached_checkpoint_v1",
            "config": tokenizer.config_dict(),
            "state_dict": tokenizer.state_dict(),
            "summary": summary,
        },
        output_dir / "checkpoint.pt",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("wrote matched frame-tokenizer outputs to %s", output_dir)


if __name__ == "__main__":
    main()
