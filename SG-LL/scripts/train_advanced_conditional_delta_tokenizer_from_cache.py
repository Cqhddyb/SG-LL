"""Train Route B advanced conditional DeltaTokenizer from a VGGT feature cache."""

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

from openpi.models_pytorch.advanced_conditional_delta_tokenizer import AdvancedConditionalDeltaTokenizer
from openpi.models_pytorch.advanced_conditional_delta_tokenizer import AdvancedConditionalDeltaTokenizerConfig
from openpi.models_pytorch.advanced_conditional_delta_tokenizer import advanced_conditional_delta_tokenizer_loss
from openpi.models_pytorch.delta_tokenizer import geometry_distance
from openpi.models_pytorch.scheme4_condition_diagnostics import roll_batch_condition

LOGGER = logging.getLogger("train_advanced_conditional_delta_tokenizer_from_cache")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-fallback", action="store_true")
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--language-token-source",
        choices=("cache", "pooled", "zeros"),
        default="cache",
        help=(
            "Use cached language_token_features by default. pooled/zeros are explicit smoke or ablation modes, "
            "not the formal Route B target."
        ),
    )
    parser.add_argument("--language-dim", type=int, default=2048, help="Used only when --language-token-source=zeros.")
    parser.add_argument(
        "--eval-condition-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report shuffled-language, shuffled-action, and zero-condition reconstruction metrics during eval.",
    )

    parser.add_argument("--num-delta-tokens", type=int, default=8)
    parser.add_argument("--delta-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--condition-dropout-prob", type=float, default=0.0)
    parser.add_argument("--relation-max-tokens", type=int, default=128)
    parser.add_argument("--no-zero-init-decoder", action="store_true")

    parser.add_argument("--smooth-l1-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--identity-weight", type=float, default=0.0)
    parser.add_argument("--negative-condition-weight", type=float, default=0.0)
    parser.add_argument("--negative-condition-margin", type=float, default=0.05)
    parser.add_argument("--negative-condition-modes", default="language,action")
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
        output_dir = Path("artifacts/advanced_conditional_delta_tokenizer_cached") / f"{cache_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _attach_language_token_features(features: FeatureCacheTensors, args: argparse.Namespace) -> FeatureCacheTensors:
    metadata = dict(features.metadata)
    sample_count = int(features.current_tokens.shape[0])
    device = features.current_tokens.device

    if args.language_token_source == "cache":
        for key in ("language_token_features", "language_token_mask"):
            if key not in metadata:
                raise ValueError(
                    "Cache metadata has no language_token_features/language_token_mask. "
                    "Run scripts/add_language_features_to_delta_cache.py --store-token-features first, "
                    "or choose --language-token-source pooled/zeros only for smoke diagnostics."
                )
        language_tokens = metadata["language_token_features"].float()
        language_mask = metadata["language_token_mask"].bool()
    elif args.language_token_source == "pooled":
        if "language_features" not in metadata:
            raise ValueError(
                "Cache metadata has no pooled language_features. Run scripts/add_language_features_to_delta_cache.py "
                "or choose --language-token-source zeros only for smoke tests."
            )
        pooled = metadata["language_features"].float()
        if pooled.ndim != 2 or pooled.shape[0] != sample_count:
            raise ValueError(f"language_features must have shape [S, D], got {tuple(pooled.shape)}")
        language_tokens = pooled[:, None, :]
        language_mask = torch.ones(sample_count, 1, dtype=torch.bool, device=pooled.device)
    else:
        if int(args.language_dim) <= 0:
            raise ValueError("--language-dim must be positive when --language-token-source=zeros")
        language_tokens = torch.zeros(
            sample_count,
            1,
            int(args.language_dim),
            dtype=torch.float32,
            device=device,
        )
        language_mask = torch.ones(sample_count, 1, dtype=torch.bool, device=device)

    if language_tokens.ndim != 3 or language_tokens.shape[0] != sample_count:
        raise ValueError(f"language_token_features must have shape [S, L, D], got {tuple(language_tokens.shape)}")
    if language_mask.shape != language_tokens.shape[:2]:
        raise ValueError(
            f"language_token_mask must have shape {tuple(language_tokens.shape[:2])}, got {tuple(language_mask.shape)}"
        )
    metadata["language_token_features"] = language_tokens.to(device=device)
    metadata["language_token_mask"] = language_mask.to(device=device)
    return FeatureCacheTensors(
        current_tokens=features.current_tokens,
        future_tokens=features.future_tokens,
        token_mask=features.token_mask,
        horizon_ids=features.horizon_ids,
        future_teacher_fallback=features.future_teacher_fallback,
        metadata=metadata,
        manifest=features.manifest,
    )


def _required_condition_tensors(
    features: FeatureCacheTensors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("actions", "state", "language_token_features", "language_token_mask"):
        if key not in features.metadata:
            raise ValueError(f"Feature cache metadata is missing required tensor {key!r}")
    actions = features.metadata["actions"].float()
    state = features.metadata["state"].float()
    language_tokens = features.metadata["language_token_features"].float()
    language_mask = features.metadata["language_token_mask"].bool()
    sample_count = int(features.current_tokens.shape[0])
    if actions.ndim != 3 or actions.shape[0] != sample_count:
        raise ValueError(f"actions must have shape [S, H, A], got {tuple(actions.shape)}")
    if state.ndim != 2 or state.shape[0] != sample_count:
        raise ValueError(f"state must have shape [S, D_state], got {tuple(state.shape)}")
    if language_tokens.ndim != 3 or language_tokens.shape[0] != sample_count:
        raise ValueError(f"language_token_features must have shape [S, L, D], got {tuple(language_tokens.shape)}")
    if language_mask.shape != language_tokens.shape[:2]:
        raise ValueError(
            f"language_token_mask must have shape {tuple(language_tokens.shape[:2])}, got {tuple(language_mask.shape)}"
        )
    return language_tokens, language_mask, actions, state


def _negative_modes(args: argparse.Namespace) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in str(args.negative_condition_modes).split(",") if mode.strip())
    invalid = set(modes) - {"language", "action"}
    if invalid:
        raise ValueError(f"--negative-condition-modes only supports language/action, got {sorted(invalid)}")
    return modes


def _tokenizer_config(
    args: argparse.Namespace,
    features: FeatureCacheTensors,
) -> AdvancedConditionalDeltaTokenizerConfig:
    language_tokens, _language_mask, actions, state = _required_condition_tensors(features)
    max_horizon = max(int(features.horizon_ids.max().detach().cpu()) + 1, 64)
    return AdvancedConditionalDeltaTokenizerConfig(
        teacher_dim=int(features.current_tokens.shape[-1]),
        language_dim=int(language_tokens.shape[-1]),
        action_horizon=int(actions.shape[1]),
        action_dim=int(actions.shape[2]),
        state_dim=int(state.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        delta_dim=int(args.delta_dim),
        num_delta_tokens=int(args.num_delta_tokens),
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        condition_dropout_prob=float(args.condition_dropout_prob),
        max_tokens=int(features.current_tokens.shape[1]),
        max_language_tokens=int(language_tokens.shape[1]),
        max_horizon=max_horizon,
        relation_max_tokens=int(args.relation_max_tokens),
        zero_init_decoder=not args.no_zero_init_decoder,
    )


def _loss_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "smooth_l1_weight": float(args.smooth_l1_weight),
        "cosine_weight": float(args.cosine_weight),
        "relation_weight": float(args.relation_weight),
        "identity_weight": float(args.identity_weight),
        "negative_condition_weight": float(args.negative_condition_weight),
        "negative_condition_margin": float(args.negative_condition_margin),
        "negative_condition_modes": _negative_modes(args),
    }


def _metric_scalars(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _subset(
    features: FeatureCacheTensors,
    indices: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> FeatureCacheTensors:
    indices = indices.to(device=features.current_tokens.device, dtype=torch.long)
    metadata = {
        key: value.index_select(0, indices)
        for key, value in features.metadata.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (features.current_tokens.shape[0],)
    }
    current_tokens = features.current_tokens.index_select(0, indices)
    future_tokens = features.future_tokens.index_select(0, indices)
    token_mask = features.token_mask.index_select(0, indices)
    horizon_ids = features.horizon_ids.index_select(0, indices)
    future_teacher_fallback = features.future_teacher_fallback.index_select(0, indices)
    if device is not None:
        current_tokens = current_tokens.to(device=device, non_blocking=True)
        future_tokens = future_tokens.to(device=device, non_blocking=True)
        token_mask = token_mask.to(device=device, non_blocking=True)
        horizon_ids = horizon_ids.to(device=device, non_blocking=True)
        future_teacher_fallback = future_teacher_fallback.to(device=device, non_blocking=True)
        metadata = {key: value.to(device=device, non_blocking=True) for key, value in metadata.items()}
    return FeatureCacheTensors(
        current_tokens=current_tokens,
        future_tokens=future_tokens,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        future_teacher_fallback=future_teacher_fallback,
        metadata=metadata,
        manifest=features.manifest,
    )


def _split_indices(
    sample_count: int,
    val_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_count < 2:
        raise ValueError("At least two samples are required for a train/val split")
    val_count = min(max(int(val_samples), 1), sample_count - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(sample_count, generator=generator)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return train_idx, val_idx


def _rec_metrics_for_conditions(
    tokenizer: AdvancedConditionalDeltaTokenizer,
    features: FeatureCacheTensors,
    *,
    language_tokens: torch.Tensor,
    language_mask: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    args: argparse.Namespace,
    prefix: str,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    output = tokenizer(
        features.current_tokens,
        features.future_tokens,
        language_token_features=language_tokens,
        language_token_mask=language_mask,
        actions=actions,
        state=state,
        token_mask=features.token_mask,
        horizon_ids=features.horizon_ids,
    )
    loss, _metrics = geometry_distance(
        output.future_recon,
        features.future_tokens,
        features.token_mask,
        smooth_l1_weight=float(args.smooth_l1_weight),
        cosine_weight=float(args.cosine_weight),
        relation_weight=float(args.relation_weight),
        relation_max_tokens=int(args.relation_max_tokens),
    )
    result = {
        f"{prefix}_rec_loss": loss.detach(),
        f"{prefix}_delta_abs_mean": output.delta_tokens.float().abs().mean().detach(),
    }
    if valid is not None:
        result[f"{prefix}_valid"] = valid.to(dtype=torch.float32).detach()
    return result


def _condition_sensitivity_metrics(
    tokenizer: AdvancedConditionalDeltaTokenizer,
    features: FeatureCacheTensors,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    language_tokens, language_mask, actions, state = _required_condition_tensors(features)
    metrics: dict[str, torch.Tensor] = {}

    neg_language, language_valid = roll_batch_condition(language_tokens)
    neg_language_mask, mask_valid = roll_batch_condition(language_mask)
    metrics.update(
        _rec_metrics_for_conditions(
            tokenizer,
            features,
            language_tokens=neg_language,
            language_mask=neg_language_mask,
            actions=actions,
            state=state,
            args=args,
            prefix="advanced_eval_shuffle_language",
            valid=language_valid & mask_valid,
        )
    )

    neg_actions, action_valid = roll_batch_condition(actions)
    metrics.update(
        _rec_metrics_for_conditions(
            tokenizer,
            features,
            language_tokens=language_tokens,
            language_mask=language_mask,
            actions=neg_actions,
            state=state,
            args=args,
            prefix="advanced_eval_shuffle_action",
            valid=action_valid,
        )
    )

    metrics.update(
        _rec_metrics_for_conditions(
            tokenizer,
            features,
            language_tokens=torch.zeros_like(language_tokens),
            language_mask=language_mask,
            actions=torch.zeros_like(actions),
            state=torch.zeros_like(state),
            args=args,
            prefix="advanced_eval_zero_condition",
        )
    )
    return metrics


def _evaluate(
    tokenizer: AdvancedConditionalDeltaTokenizer,
    features: FeatureCacheTensors,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    weighted: dict[str, float] = {}
    total = int(indices.shape[0])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row = _subset(features, indices[start:end], device=device)
            language_tokens, language_mask, actions, state = _required_condition_tensors(row)
            loss_output = advanced_conditional_delta_tokenizer_loss(
                tokenizer,
                row.current_tokens,
                row.future_tokens,
                language_token_features=language_tokens,
                language_token_mask=language_mask,
                actions=actions,
                state=state,
                token_mask=row.token_mask,
                horizon_ids=row.horizon_ids,
                **_loss_kwargs(args),
            )
            metrics = dict(loss_output.metrics)
            if bool(args.eval_condition_sensitivity):
                metrics.update(_condition_sensitivity_metrics(tokenizer, row, args))
            scalars = _metric_scalars(metrics)
            count = end - start
            for key, value in scalars.items():
                weighted[key] = weighted.get(key, 0.0) + value * count
    return {key: value / total for key, value in weighted.items()}


def _baseline_current_as_future(
    features: FeatureCacheTensors,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    weighted: dict[str, float] = {}
    total = int(indices.shape[0])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row = _subset(features, indices[start:end], device=device)
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
    tokenizer: AdvancedConditionalDeltaTokenizer,
    features: FeatureCacheTensors,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    train_count = int(train_idx.shape[0])
    batch_size = min(int(args.batch_size), train_count)
    rows: list[dict[str, float]] = []
    metrics_path = output_dir / "metrics.jsonl"

    tokenizer.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(int(args.train_steps)):
            batch_pos = torch.randint(train_count, (batch_size,), generator=generator)
            batch_rows = train_idx.index_select(0, batch_pos)
            batch = _subset(features, batch_rows, device=device)
            language_tokens, language_mask, actions, state = _required_condition_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            loss_output = advanced_conditional_delta_tokenizer_loss(
                tokenizer,
                batch.current_tokens,
                batch.future_tokens,
                language_token_features=language_tokens,
                language_token_mask=language_mask,
                actions=actions,
                state=state,
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
                    row["advanced_conditional_delta_tokenizer_loss"],
                    row["advanced_conditional_delta_rec_loss"],
                )

    final_train = _evaluate(tokenizer, features, train_idx, args, batch_size=batch_size, device=device)
    final_val = _evaluate(
        tokenizer,
        features,
        val_idx,
        args,
        batch_size=max(1, min(int(args.batch_size), int(val_idx.shape[0]))),
        device=device,
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
        device=torch.device("cpu"),
        max_samples=args.max_samples,
        drop_fallback=not args.keep_fallback,
    )
    features = _attach_language_token_features(features, args)
    train_idx, val_idx = _split_indices(
        int(features.current_tokens.shape[0]),
        int(args.val_samples),
        int(args.seed),
    )
    tokenizer_config = _tokenizer_config(args, features)
    tokenizer = AdvancedConditionalDeltaTokenizer(tokenizer_config).to(device)
    write_json(output_dir / "tokenizer_config.json", tokenizer.config_dict())

    train_batch_size = min(int(args.batch_size), int(train_idx.shape[0]))
    val_batch_size = max(1, min(int(args.batch_size), int(val_idx.shape[0])))
    initial_train = _evaluate(tokenizer, features, train_idx, args, batch_size=train_batch_size, device=device)
    initial_val = _evaluate(tokenizer, features, val_idx, args, batch_size=val_batch_size, device=device)
    baseline_train = _baseline_current_as_future(features, train_idx, args, batch_size=train_batch_size, device=device)
    baseline_val = _baseline_current_as_future(features, val_idx, args, batch_size=val_batch_size, device=device)
    LOGGER.info(
        "Initial train loss=%.6f val loss=%.6f",
        initial_train["advanced_conditional_delta_tokenizer_loss"],
        initial_val["advanced_conditional_delta_tokenizer_loss"],
    )

    rows, final_train, final_val = _train(
        tokenizer,
        features,
        train_idx,
        val_idx,
        args,
        output_dir,
        device=device,
    )
    language_tokens, language_mask, actions, state = _required_condition_tensors(features)
    summary: dict[str, Any] = {
        "schema_version": "scheme4_advanced_conditional_delta_tokenizer_cached_train_v1",
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
                "metadata_fields",
                "stores_language_token_features",
            )
        },
        "feature_shapes": feature_cache_shape_summary(features),
        "condition_shapes": {
            "language_token_features": list(language_tokens.shape),
            "language_token_mask": list(language_mask.shape),
            "actions": list(actions.shape),
            "state": list(state.shape),
        },
        "train_indices": [int(x) for x in train_idx.detach().cpu().tolist()],
        "val_indices": [int(x) for x in val_idx.detach().cpu().tolist()],
        "tokenizer_config": tokenizer.config_dict(),
        "baseline_train_current_as_future": baseline_train,
        "baseline_val_current_as_future": baseline_val,
        "initial_train_metrics": initial_train,
        "initial_val_metrics": initial_val,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "train_loss_decreased": final_train["advanced_conditional_delta_tokenizer_loss"]
        < initial_train["advanced_conditional_delta_tokenizer_loss"],
        "train_rec_loss_decreased": final_train["advanced_conditional_delta_rec_loss"]
        < initial_train["advanced_conditional_delta_rec_loss"],
        "val_loss_decreased": final_val["advanced_conditional_delta_tokenizer_loss"]
        < initial_val["advanced_conditional_delta_tokenizer_loss"],
        "val_rec_loss_decreased": final_val["advanced_conditional_delta_rec_loss"]
        < initial_val["advanced_conditional_delta_rec_loss"],
        "last_train_row": rows[-1] if rows else None,
        "cuda_max_memory_allocated_bytes": None
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "schema_version": "scheme4_advanced_conditional_delta_tokenizer_cached_checkpoint_v1",
            "config": tokenizer.config_dict(),
            "state_dict": tokenizer.state_dict(),
            "summary": summary,
        },
        output_dir / "checkpoint.pt",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote advanced conditional tokenizer outputs to %s", output_dir)


if __name__ == "__main__":
    main()
