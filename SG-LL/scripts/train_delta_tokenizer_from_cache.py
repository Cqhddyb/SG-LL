"""Train the standalone scheme3 DeltaTokenizer from a VGGT feature cache."""

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

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizer
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import delta_tokenizer_loss
from openpi.models_pytorch.delta_tokenizer import geometry_distance

LOGGER = logging.getLogger("train_delta_tokenizer_from_cache")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-fallback", action="store_true")
    parser.add_argument("--val-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--num-delta-tokens", type=int, default=8)
    parser.add_argument("--delta-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--relation-max-tokens", type=int, default=128)
    parser.add_argument("--no-zero-init-decoder", action="store_true")

    parser.add_argument("--smooth-l1-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--identity-weight", type=float, default=0.05)
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


def _resolve_output_dir(output_dir_arg: str | None, cache_dir: Path) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts/delta_tokenizer_cached") / f"{cache_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _tokenizer_config(args: argparse.Namespace, features: FeatureCacheTensors) -> DeltaTokenizerConfig:
    max_horizon = max(int(features.horizon_ids.max().detach().cpu()) + 1, 64)
    return DeltaTokenizerConfig(
        teacher_dim=int(features.current_tokens.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        delta_dim=int(args.delta_dim),
        num_delta_tokens=int(args.num_delta_tokens),
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        max_tokens=int(features.current_tokens.shape[1]),
        max_horizon=max_horizon,
        relation_max_tokens=int(args.relation_max_tokens),
        zero_init_decoder=not args.no_zero_init_decoder,
    )


def _loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "smooth_l1_weight": float(args.smooth_l1_weight),
        "cosine_weight": float(args.cosine_weight),
        "relation_weight": float(args.relation_weight),
        "identity_weight": float(args.identity_weight),
    }


def _metric_scalars(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _subset(features: FeatureCacheTensors, indices: torch.Tensor) -> FeatureCacheTensors:
    indices = indices.to(device=features.current_tokens.device, dtype=torch.long)
    metadata = {
        key: value.index_select(0, indices)
        for key, value in features.metadata.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (features.current_tokens.shape[0],)
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


def _split_indices(
    sample_count: int, val_samples: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_count < 2:
        raise ValueError("At least two samples are required for a train/val cache smoke")
    val_count = min(max(int(val_samples), 1), sample_count - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(sample_count, generator=generator)
    val_idx = perm[:val_count].to(device=device)
    train_idx = perm[val_count:].to(device=device)
    return train_idx, val_idx


def _evaluate(
    tokenizer: DeltaTokenizer,
    features: FeatureCacheTensors,
    args: argparse.Namespace,
    *,
    batch_size: int,
) -> dict[str, float]:
    tokenizer.eval()
    weighted: dict[str, float] = {}
    total = int(features.current_tokens.shape[0])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row = _subset(features, torch.arange(start, end, device=features.current_tokens.device))
            loss_output = delta_tokenizer_loss(
                tokenizer,
                row.current_tokens,
                row.future_tokens,
                token_mask=row.token_mask,
                horizon_ids=row.horizon_ids,
                **_loss_kwargs(args),
            )
            scalars = _metric_scalars(loss_output.metrics)
            count = end - start
            for key, value in scalars.items():
                weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / total for key, value in weighted.items()}


def _baseline_current_as_future(
    features: FeatureCacheTensors,
    args: argparse.Namespace,
    *,
    batch_size: int,
) -> dict[str, float]:
    weighted: dict[str, float] = {}
    total = int(features.current_tokens.shape[0])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row = _subset(features, torch.arange(start, end, device=features.current_tokens.device))
            loss, metrics = geometry_distance(
                row.current_tokens,
                row.future_tokens,
                row.token_mask,
                smooth_l1_weight=float(args.smooth_l1_weight),
                cosine_weight=float(args.cosine_weight),
                relation_weight=float(args.relation_weight),
                relation_max_tokens=int(args.relation_max_tokens),
            )
            scalars = _metric_scalars(metrics)
            scalars["current_as_future_loss"] = float(loss.detach().cpu())
            count = end - start
            for key, value in scalars.items():
                weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / total for key, value in weighted.items()}


def _train(
    tokenizer: DeltaTokenizer,
    train_features: FeatureCacheTensors,
    val_features: FeatureCacheTensors,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    train_count = int(train_features.current_tokens.shape[0])
    batch_size = min(int(args.batch_size), train_count)
    rows: list[dict[str, float]] = []
    metrics_path = output_dir / "metrics.jsonl"

    tokenizer.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(int(args.train_steps)):
            batch_idx = torch.randint(train_count, (batch_size,), generator=generator)
            batch = _subset(train_features, batch_idx.to(device=train_features.current_tokens.device))
            optimizer.zero_grad(set_to_none=True)
            loss_output = delta_tokenizer_loss(
                tokenizer,
                batch.current_tokens,
                batch.future_tokens,
                token_mask=batch.token_mask,
                horizon_ids=batch.horizon_ids,
                **_loss_kwargs(args),
            )
            if not torch.isfinite(loss_output.loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss_output.loss.detach().cpu()}")
            loss_output.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()

            row = {
                "step": float(step + 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()),
                **_metric_scalars(loss_output.metrics),
            }
            rows.append(row)
            metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (step + 1) % int(args.log_every) == 0 or step == 0:
                LOGGER.info(
                    "step=%s train_loss=%.6f train_rec=%.6f",
                    step + 1,
                    row["delta_tokenizer_loss"],
                    row["delta_rec_loss"],
                )

    final_train = _evaluate(tokenizer, train_features, args, batch_size=batch_size)
    final_val = _evaluate(
        tokenizer,
        val_features,
        args,
        batch_size=max(1, min(int(args.batch_size), int(val_features.current_tokens.shape[0]))),
    )
    return rows, final_train, final_val


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive")

    cache_dir = Path(args.cache_dir)
    device = _select_device(args.device)
    _set_seed(int(args.seed), device)
    output_dir = _resolve_output_dir(args.output_dir, cache_dir)
    write_json(output_dir / "args.json", vars(args))

    features = load_feature_cache(
        cache_dir,
        device=device,
        max_samples=args.max_samples,
        drop_fallback=not args.keep_fallback,
    )
    train_idx, val_idx = _split_indices(
        int(features.current_tokens.shape[0]), int(args.val_samples), int(args.seed), device
    )
    train_features = _subset(features, train_idx)
    val_features = _subset(features, val_idx)
    tokenizer_config = _tokenizer_config(args, features)
    tokenizer = DeltaTokenizer(tokenizer_config).to(device)
    write_json(output_dir / "tokenizer_config.json", tokenizer.config_dict())

    train_batch_size = min(int(args.batch_size), int(train_features.current_tokens.shape[0]))
    val_batch_size = max(1, min(int(args.batch_size), int(val_features.current_tokens.shape[0])))
    initial_train = _evaluate(tokenizer, train_features, args, batch_size=train_batch_size)
    initial_val = _evaluate(tokenizer, val_features, args, batch_size=val_batch_size)
    baseline_train = _baseline_current_as_future(train_features, args, batch_size=train_batch_size)
    baseline_val = _baseline_current_as_future(val_features, args, batch_size=val_batch_size)
    LOGGER.info(
        "Initial train loss=%.6f val loss=%.6f",
        initial_train["delta_tokenizer_loss"],
        initial_val["delta_tokenizer_loss"],
    )

    rows, final_train, final_val = _train(tokenizer, train_features, val_features, args, output_dir)
    summary: dict[str, Any] = {
        "schema_version": "scheme3_delta_tokenizer_cached_train_v1",
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "cache_dir": str(cache_dir),
        "device": str(device),
        "args": vars(args),
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
        "train_indices": [int(x) for x in train_idx.detach().cpu().tolist()],
        "val_indices": [int(x) for x in val_idx.detach().cpu().tolist()],
        "tokenizer_config": tokenizer.config_dict(),
        "baseline_train_current_as_future": baseline_train,
        "baseline_val_current_as_future": baseline_val,
        "initial_train_metrics": initial_train,
        "initial_val_metrics": initial_val,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "train_loss_decreased": final_train["delta_tokenizer_loss"] < initial_train["delta_tokenizer_loss"],
        "train_rec_loss_decreased": final_train["delta_rec_loss"] < initial_train["delta_rec_loss"],
        "val_loss_decreased": final_val["delta_tokenizer_loss"] < initial_val["delta_tokenizer_loss"],
        "val_rec_loss_decreased": final_val["delta_rec_loss"] < initial_val["delta_rec_loss"],
        "last_train_row": rows[-1] if rows else None,
        "cuda_max_memory_allocated_bytes": None
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "schema_version": "scheme3_delta_tokenizer_cached_checkpoint_v1",
            "config": tokenizer.config_dict(),
            "state_dict": tokenizer.state_dict(),
            "summary": summary,
        },
        output_dir / "checkpoint.pt",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote cached tokenizer outputs to %s", output_dir)


if __name__ == "__main__":
    main()
