"""Train the standalone scheme3 GT-action-conditioned delta predictor.

This is an auxiliary diagnostic only.  It freezes a trained DeltaTokenizer,
uses the tokenizer encoder to produce target delta tokens from cached
current/future VGGT features, and trains a separate DeltaActionPredictor from
current tokens plus cached action/state conditions.  By default the action
condition is the cached GT action chunk; with --action-source estimated it is a
stop-gradient endpoint action estimate stored in the cache.  No predicted delta
is fed into the pi0.5 action path.

Use --condition-mode to run auxiliary ablations. Disabled conditions are
replaced by same-shaped zeros before they enter the predictor, while the frozen
decoder still receives the real current tokens for future-reconstruction
diagnostics.
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

from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictor
from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictorConfig
from openpi.models_pytorch.delta_action_predictor import delta_action_predictor_loss
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizer
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import geometry_distance

LOGGER = logging.getLogger("train_delta_action_predictor_from_cache")

DEFAULT_CACHE_DIR = "artifacts/delta_vggt_cache/s3_m2b_cache_b2_target512_shuffle_seed13"
DEFAULT_TOKENIZER_CHECKPOINT = (
    "artifacts/delta_tokenizer_cached/s3_m1b_cached_n507_steps2000_lr3e4_noid_seed13/checkpoint.pt"
)

CONDITION_MODE_FLAGS = {
    "full": {"current": True, "action": True, "state": True},
    "current_only": {"current": True, "action": False, "state": False},
    "state_only": {"current": False, "action": False, "state": True},
    "action_only": {"current": False, "action": True, "state": False},
    "current_state": {"current": True, "action": False, "state": True},
    "current_action": {"current": True, "action": True, "state": False},
    "state_action": {"current": False, "action": True, "state": True},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokenizer-checkpoint", default=DEFAULT_TOKENIZER_CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-fallback", action="store_true")
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precompute-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--condition-mode",
        choices=sorted(CONDITION_MODE_FLAGS),
        default="full",
        help="Which cached conditions the predictor may use. Disabled conditions are replaced by explicit zeros.",
    )
    parser.add_argument(
        "--action-source",
        choices=("gt", "estimated"),
        default="gt",
        help=(
            "Action tensor used as the predictor condition. "
            "gt reads cache metadata 'actions'; estimated reads 'estimated_actions'."
        ),
    )

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--zero-init-output", action="store_true")

    parser.add_argument("--delta-smooth-l1-weight", type=float, default=1.0)
    parser.add_argument("--delta-mse-weight", type=float, default=0.0)
    parser.add_argument("--delta-cosine-weight", type=float, default=0.1)
    parser.add_argument("--future-smooth-l1-weight", type=float, default=1.0)
    parser.add_argument("--future-cosine-weight", type=float, default=0.1)
    parser.add_argument("--future-relation-weight", type=float, default=0.0)
    parser.add_argument("--future-relation-max-tokens", type=int, default=128)
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
        output_dir = Path("artifacts/delta_action_predictor") / f"{cache_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _split_indices(
    sample_count: int, val_samples: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_count < 2:
        raise ValueError("At least two samples are required for a train/val predictor diagnostic")
    val_count = min(max(int(val_samples), 1), sample_count - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(sample_count, generator=generator)
    val_idx = perm[:val_count].to(device=device)
    train_idx = perm[val_count:].to(device=device)
    return train_idx, val_idx


def _metric_scalars(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "smooth_l1_weight": float(args.delta_smooth_l1_weight),
        "mse_weight": float(args.delta_mse_weight),
        "cosine_weight": float(args.delta_cosine_weight),
    }


def _future_loss_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "smooth_l1_weight": float(args.future_smooth_l1_weight),
        "cosine_weight": float(args.future_cosine_weight),
        "relation_weight": float(args.future_relation_weight),
        "relation_max_tokens": int(args.future_relation_max_tokens),
    }


def _metadata_shape_summary(features: FeatureCacheTensors) -> dict[str, list[int]]:
    return {
        key: list(value.shape) for key, value in sorted(features.metadata.items()) if isinstance(value, torch.Tensor)
    }


def _action_key(action_source: str) -> str:
    if action_source == "gt":
        return "actions"
    if action_source == "estimated":
        return "estimated_actions"
    raise ValueError(f"Unknown action source {action_source!r}; expected 'gt' or 'estimated'")


def _require_action_state(
    features: FeatureCacheTensors, *, action_source: str
) -> tuple[torch.Tensor, torch.Tensor, str]:
    action_key = _action_key(action_source)
    missing = [key for key in (action_key, "state") if key not in features.metadata]
    if missing:
        raise ValueError(
            f"Feature cache is missing required auxiliary tensor(s) {missing}; "
            "rebuild or augment the cache with the requested action source and state"
        )
    actions = features.metadata[action_key]
    state = features.metadata["state"]
    sample_count = int(features.current_tokens.shape[0])
    if actions.ndim != 3 or actions.shape[0] != sample_count:
        raise ValueError(f"{action_key} must have shape [S, H, A], got {tuple(actions.shape)}")
    if state.ndim != 2 or state.shape[0] != sample_count:
        raise ValueError(f"state must have shape [S, D_state], got {tuple(state.shape)}")
    return actions.float(), state.float(), action_key


def _condition_flags(mode: str) -> dict[str, bool]:
    if mode not in CONDITION_MODE_FLAGS:
        raise ValueError(f"Unknown condition mode {mode!r}; expected one of {sorted(CONDITION_MODE_FLAGS)}")
    return dict(CONDITION_MODE_FLAGS[mode])


def _apply_condition_mode(
    current_tokens: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flags = _condition_flags(mode)
    condition_current = current_tokens if flags["current"] else torch.zeros_like(current_tokens)
    condition_actions = actions if flags["action"] else torch.zeros_like(actions)
    condition_state = state if flags["state"] else torch.zeros_like(state)
    return condition_current, condition_actions, condition_state


def _load_frozen_tokenizer(checkpoint_path: Path, device: torch.device) -> DeltaTokenizer:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError(f"Tokenizer checkpoint {checkpoint_path} is missing config/state_dict")
    tokenizer = DeltaTokenizer(DeltaTokenizerConfig(**payload["config"]))
    tokenizer.load_state_dict(payload["state_dict"])
    tokenizer.to(device=device)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(requires_grad=False)
    return tokenizer


def _validate_tokenizer_cache_match(tokenizer: DeltaTokenizer, features: FeatureCacheTensors) -> None:
    if int(features.current_tokens.shape[-1]) != tokenizer.config.teacher_dim:
        raise ValueError(
            "Cache teacher dim does not match tokenizer config: "
            f"{int(features.current_tokens.shape[-1])} vs {tokenizer.config.teacher_dim}"
        )
    if int(features.current_tokens.shape[1]) > tokenizer.config.max_tokens:
        raise ValueError(
            "Cache token count exceeds tokenizer max_tokens: "
            f"{int(features.current_tokens.shape[1])} vs {tokenizer.config.max_tokens}"
        )
    if int(features.horizon_ids.max().detach().cpu()) >= tokenizer.config.max_horizon:
        raise ValueError(
            "Cache horizon id exceeds tokenizer max_horizon: "
            f"{int(features.horizon_ids.max().detach().cpu())} vs {tokenizer.config.max_horizon}"
        )


def _predictor_config(
    args: argparse.Namespace,
    features: FeatureCacheTensors,
    actions: torch.Tensor,
    state: torch.Tensor,
    tokenizer: DeltaTokenizer,
) -> DeltaActionPredictorConfig:
    max_horizon = max(int(features.horizon_ids.max().detach().cpu()) + 1, int(tokenizer.config.max_horizon), 64)
    return DeltaActionPredictorConfig(
        teacher_dim=int(features.current_tokens.shape[-1]),
        delta_dim=int(tokenizer.config.delta_dim),
        num_delta_tokens=int(tokenizer.config.num_delta_tokens),
        action_horizon=int(actions.shape[1]),
        action_dim=int(actions.shape[2]),
        state_dim=int(state.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        max_horizon=max_horizon,
        zero_init_output=bool(args.zero_init_output),
    )


def _precompute_target_delta(
    tokenizer: DeltaTokenizer,
    features: FeatureCacheTensors,
    *,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("--precompute-batch-size must be positive")
    chunks: list[torch.Tensor] = []
    sample_count = int(features.current_tokens.shape[0])
    device = features.current_tokens.device
    tokenizer.eval()
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            idx = torch.arange(start, end, device=device)
            delta = tokenizer.encode(
                features.current_tokens.index_select(0, idx),
                features.future_tokens.index_select(0, idx),
                token_mask=features.token_mask.index_select(0, idx),
                horizon_ids=features.horizon_ids.index_select(0, idx),
            )
            chunks.append(delta.detach())
    return torch.cat(chunks, dim=0)


def _target_delta_stats(target_delta: torch.Tensor) -> dict[str, float | list[int]]:
    return {
        "shape": list(target_delta.shape),
        "mean": float(target_delta.float().mean().detach().cpu()),
        "std": float(target_delta.float().std(unbiased=False).detach().cpu()),
        "l1_mean": float(target_delta.float().abs().mean().detach().cpu()),
        "l2_mean": float(target_delta.float().square().sum(dim=-1).sqrt().mean().detach().cpu()),
    }


def _make_shuffled_action_indices(indices: torch.Tensor, *, seed: int) -> torch.Tensor:
    if int(indices.numel()) < 2:
        return indices
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(int(indices.numel()), generator=generator).to(device=indices.device)
    if torch.equal(perm, torch.arange(int(indices.numel()), device=indices.device)):
        perm = perm.roll(1)
    return indices.index_select(0, perm)


def _weighted_update(weighted: dict[str, float], metrics: dict[str, float], count: int) -> None:
    for key, value in metrics.items():
        weighted[key] = weighted.get(key, 0.0) + float(value) * count


def _evaluate_predictor(
    predictor: DeltaActionPredictor,
    tokenizer: DeltaTokenizer,
    features: FeatureCacheTensors,
    actions: torch.Tensor,
    state: torch.Tensor,
    target_delta: torch.Tensor,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    batch_size: int,
    shuffle_actions: bool,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    predictor.eval()
    weighted: dict[str, float] = {}
    total = int(indices.numel())
    if total <= 0:
        raise ValueError("Cannot evaluate an empty split")
    shuffled_action_indices = _make_shuffled_action_indices(indices, seed=int(args.seed) + 7919)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row_idx = indices[start:end]
            action_idx = shuffled_action_indices[start:end] if shuffle_actions else row_idx
            current = features.current_tokens.index_select(0, row_idx)
            row_actions = actions.index_select(0, action_idx)
            row_state = state.index_select(0, row_idx)
            condition_current, condition_actions, condition_state = _apply_condition_mode(
                current,
                row_actions,
                row_state,
                mode=str(args.condition_mode),
            )
            token_mask = features.token_mask.index_select(0, row_idx)
            horizon_ids = features.horizon_ids.index_select(0, row_idx)
            loss_output = delta_action_predictor_loss(
                predictor,
                condition_current,
                condition_actions,
                target_delta.index_select(0, row_idx),
                state=condition_state,
                token_mask=token_mask,
                horizon_ids=horizon_ids,
                **_loss_kwargs(args),
            )
            pred_future = tokenizer.decode(
                current,
                loss_output.pred_delta_tokens,
                token_mask=token_mask,
                horizon_ids=horizon_ids,
            )
            future_loss, future_metrics = geometry_distance(
                pred_future,
                features.future_tokens.index_select(0, row_idx).detach(),
                token_mask,
                **_future_loss_kwargs(args),
            )
            metrics = _metric_scalars(loss_output.metrics)
            metrics["future_rec_loss"] = float(future_loss.detach().cpu())
            metrics.update({f"future_{key}": value for key, value in _metric_scalars(future_metrics).items()})
            _weighted_update(weighted, metrics, end - start)
    return {key: value / total for key, value in weighted.items()}


def _evaluate_geometry_baselines(
    tokenizer: DeltaTokenizer,
    features: FeatureCacheTensors,
    target_delta: torch.Tensor,
    indices: torch.Tensor,
    args: argparse.Namespace,
    *,
    batch_size: int,
) -> dict[str, float]:
    weighted: dict[str, float] = {}
    total = int(indices.numel())
    device = features.current_tokens.device
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            row_idx = indices[start:end]
            current = features.current_tokens.index_select(0, row_idx)
            future = features.future_tokens.index_select(0, row_idx)
            token_mask = features.token_mask.index_select(0, row_idx)
            horizon_ids = features.horizon_ids.index_select(0, row_idx)

            current_loss, current_metrics = geometry_distance(
                current,
                future,
                token_mask,
                **_future_loss_kwargs(args),
            )
            oracle_future = tokenizer.decode(
                current,
                target_delta.index_select(0, row_idx),
                token_mask=token_mask,
                horizon_ids=horizon_ids,
            )
            oracle_loss, oracle_metrics = geometry_distance(
                oracle_future,
                future.detach(),
                token_mask,
                **_future_loss_kwargs(args),
            )
            zero = torch.zeros(
                current.shape[0],
                tokenizer.config.num_delta_tokens,
                tokenizer.config.delta_dim,
                device=device,
                dtype=target_delta.dtype,
            )
            zero_future = tokenizer.decode(current, zero, token_mask=token_mask, horizon_ids=horizon_ids)
            zero_loss, zero_metrics = geometry_distance(
                zero_future,
                future.detach(),
                token_mask,
                **_future_loss_kwargs(args),
            )
            metrics = {"current_as_future_loss": float(current_loss.detach().cpu())}
            metrics.update(
                {f"current_as_future_{key}": value for key, value in _metric_scalars(current_metrics).items()}
            )
            metrics["oracle_future_rec_loss"] = float(oracle_loss.detach().cpu())
            metrics.update({f"oracle_future_{key}": value for key, value in _metric_scalars(oracle_metrics).items()})
            metrics["zero_delta_future_rec_loss"] = float(zero_loss.detach().cpu())
            metrics.update({f"zero_delta_future_{key}": value for key, value in _metric_scalars(zero_metrics).items()})
            _weighted_update(weighted, metrics, end - start)
    return {key: value / total for key, value in weighted.items()}


def _train(
    predictor: DeltaActionPredictor,
    features: FeatureCacheTensors,
    actions: torch.Tensor,
    state: torch.Tensor,
    target_delta: torch.Tensor,
    train_idx: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    train_count = int(train_idx.numel())
    batch_size = min(int(args.batch_size), train_count)
    rows: list[dict[str, float]] = []
    metrics_path = output_dir / "metrics.jsonl"

    predictor.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(int(args.train_steps)):
            local_idx = torch.randint(train_count, (batch_size,), generator=generator).to(device=train_idx.device)
            row_idx = train_idx.index_select(0, local_idx)
            current = features.current_tokens.index_select(0, row_idx)
            row_actions = actions.index_select(0, row_idx)
            row_state = state.index_select(0, row_idx)
            condition_current, condition_actions, condition_state = _apply_condition_mode(
                current,
                row_actions,
                row_state,
                mode=str(args.condition_mode),
            )
            optimizer.zero_grad(set_to_none=True)
            loss_output = delta_action_predictor_loss(
                predictor,
                condition_current,
                condition_actions,
                target_delta.index_select(0, row_idx),
                state=condition_state,
                token_mask=features.token_mask.index_select(0, row_idx),
                horizon_ids=features.horizon_ids.index_select(0, row_idx),
                **_loss_kwargs(args),
            )
            if not torch.isfinite(loss_output.loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss_output.loss.detach().cpu()}")
            loss_output.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=float(args.grad_clip_norm))
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
                    "step=%s train_delta_loss=%.6f train_delta_mse=%.6f",
                    step + 1,
                    row["delta_action_loss"],
                    row["delta_mse_loss"],
                )
    return rows


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive")

    cache_dir = Path(args.cache_dir)
    tokenizer_checkpoint = Path(args.tokenizer_checkpoint)
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
    actions, state, action_key = _require_action_state(features, action_source=str(args.action_source))
    tokenizer = _load_frozen_tokenizer(tokenizer_checkpoint, device)
    _validate_tokenizer_cache_match(tokenizer, features)
    target_delta = _precompute_target_delta(tokenizer, features, batch_size=int(args.precompute_batch_size))
    predictor_config = _predictor_config(args, features, actions, state, tokenizer)
    predictor = DeltaActionPredictor(predictor_config).to(device=device)
    write_json(output_dir / "predictor_config.json", predictor.config_dict())

    train_idx, val_idx = _split_indices(
        int(features.current_tokens.shape[0]), int(args.val_samples), int(args.seed), device
    )
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    eval_batch_size = max(1, eval_batch_size)

    train_baselines = _evaluate_geometry_baselines(
        tokenizer,
        features,
        target_delta,
        train_idx,
        args,
        batch_size=min(eval_batch_size, int(train_idx.numel())),
    )
    val_baselines = _evaluate_geometry_baselines(
        tokenizer,
        features,
        target_delta,
        val_idx,
        args,
        batch_size=min(eval_batch_size, int(val_idx.numel())),
    )
    initial_train = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        train_idx,
        args,
        batch_size=min(eval_batch_size, int(train_idx.numel())),
        shuffle_actions=False,
    )
    initial_val = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        val_idx,
        args,
        batch_size=min(eval_batch_size, int(val_idx.numel())),
        shuffle_actions=False,
    )
    LOGGER.info(
        "Initial train delta loss=%.6f val delta loss=%.6f",
        initial_train["delta_action_loss"],
        initial_val["delta_action_loss"],
    )

    rows = _train(predictor, features, actions, state, target_delta, train_idx, args, output_dir)

    final_train = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        train_idx,
        args,
        batch_size=min(eval_batch_size, int(train_idx.numel())),
        shuffle_actions=False,
    )
    final_val = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        val_idx,
        args,
        batch_size=min(eval_batch_size, int(val_idx.numel())),
        shuffle_actions=False,
    )
    final_train_shuffled = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        train_idx,
        args,
        batch_size=min(eval_batch_size, int(train_idx.numel())),
        shuffle_actions=True,
    )
    final_val_shuffled = _evaluate_predictor(
        predictor,
        tokenizer,
        features,
        actions,
        state,
        target_delta,
        val_idx,
        args,
        batch_size=min(eval_batch_size, int(val_idx.numel())),
        shuffle_actions=True,
    )

    summary: dict[str, Any] = {
        "schema_version": "scheme3_delta_action_predictor_train_v1",
        "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "cache_dir": str(cache_dir),
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "device": str(device),
        "args": vars(args),
        "condition_mode": str(args.condition_mode),
        "condition_flags": _condition_flags(str(args.condition_mode)),
        "action_source": str(args.action_source),
        "action_key": action_key,
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
        "feature_shapes": {
            **feature_cache_shape_summary(features),
            "metadata": _metadata_shape_summary(features),
        },
        "train_indices": [int(x) for x in train_idx.detach().cpu().tolist()],
        "val_indices": [int(x) for x in val_idx.detach().cpu().tolist()],
        "tokenizer_config": tokenizer.config_dict(),
        "predictor_config": predictor.config_dict(),
        "target_delta_stats": _target_delta_stats(target_delta),
        "train_geometry_baselines": train_baselines,
        "val_geometry_baselines": val_baselines,
        "initial_train_metrics": initial_train,
        "initial_val_metrics": initial_val,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "final_train_shuffled_action_metrics": final_train_shuffled,
        "final_val_shuffled_action_metrics": final_val_shuffled,
        "train_delta_loss_decreased": final_train["delta_action_loss"] < initial_train["delta_action_loss"],
        "val_delta_loss_decreased": final_val["delta_action_loss"] < initial_val["delta_action_loss"],
        "train_future_loss_decreased": final_train["future_rec_loss"] < initial_train["future_rec_loss"],
        "val_future_loss_decreased": final_val["future_rec_loss"] < initial_val["future_rec_loss"],
        "final_val_future_beats_current_as_future": final_val["future_rec_loss"]
        < val_baselines["current_as_future_loss"],
        "final_val_future_beats_zero_delta": final_val["future_rec_loss"] < val_baselines["zero_delta_future_rec_loss"],
        "final_val_future_beats_initial": final_val["future_rec_loss"] < initial_val["future_rec_loss"],
        "val_shuffled_delta_worse": final_val_shuffled["delta_action_loss"] > final_val["delta_action_loss"],
        "val_shuffled_future_worse": final_val_shuffled["future_rec_loss"] > final_val["future_rec_loss"],
        "last_train_row": rows[-1] if rows else None,
        "cuda_max_memory_allocated_bytes": None
        if device.type != "cuda"
        else int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "schema_version": "scheme3_delta_action_predictor_checkpoint_v1",
            "predictor_config": predictor.config_dict(),
            "state_dict": predictor.state_dict(),
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
            "summary": summary,
        },
        output_dir / "checkpoint.pt",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote delta action predictor outputs to %s", output_dir)


if __name__ == "__main__":
    main()
