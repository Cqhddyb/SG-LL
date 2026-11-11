"""Matched probes and interventions for auditing transition-token semantics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.delta_tokenizer import DeltaDecoder
from openpi.models_pytorch.delta_tokenizer import DeltaEncoder
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig

FutureProbeMode = Literal["current_only", "transition_only", "current_transition"]
TokenProbeMode = Literal["current_only", "future_only", "current_future"]
SwapRelation = Literal["same_primary", "different_primary", "same_primary_different_secondary"]

FUTURE_PROBE_MODES: tuple[FutureProbeMode, ...] = (
    "current_only",
    "transition_only",
    "current_transition",
)
TOKEN_PROBE_MODES: tuple[TokenProbeMode, ...] = (
    "current_only",
    "future_only",
    "current_future",
)
SWAP_RELATIONS: tuple[SwapRelation, ...] = (
    "same_primary",
    "different_primary",
    "same_primary_different_secondary",
)


def _check_rank3(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B, N, D], got {tuple(value.shape)}")


def _check_feature_pair(current_tokens: torch.Tensor, future_tokens: torch.Tensor) -> None:
    _check_rank3("current_tokens", current_tokens)
    _check_rank3("future_tokens", future_tokens)
    if current_tokens.shape != future_tokens.shape:
        raise ValueError(
            "current_tokens and future_tokens must have identical shape, "
            f"got {tuple(current_tokens.shape)} and {tuple(future_tokens.shape)}"
        )
    if current_tokens.device != future_tokens.device:
        raise ValueError(
            f"current_tokens device {current_tokens.device} must match future_tokens device {future_tokens.device}"
        )


def _check_delta_tokens(delta_tokens: torch.Tensor, config: DeltaTokenizerConfig, batch_size: int) -> None:
    _check_rank3("delta_tokens", delta_tokens)
    expected = (batch_size, config.num_delta_tokens, config.delta_dim)
    if delta_tokens.shape != expected:
        raise ValueError(f"delta_tokens must have shape {expected}, got {tuple(delta_tokens.shape)}")


class MatchedFutureDecoderProbe(nn.Module):
    """Decode future features under a controlled current/token input mode.

    Inputs:
    - current_tokens: [B, N, D_teacher]
    - delta_tokens: [B, M_delta, D_delta]
    - token_mask: optional [B, N]
    - horizon_ids: optional [B]

    Output:
    - future_recon: [B, N, D_teacher]
    """

    def __init__(self, config: DeltaTokenizerConfig, mode: FutureProbeMode) -> None:
        super().__init__()
        if mode not in FUTURE_PROBE_MODES:
            raise ValueError(f"mode must be one of {FUTURE_PROBE_MODES}, got {mode!r}")
        self.config = config
        self.mode = mode
        self.decoder = DeltaDecoder(config)

    def forward(
        self,
        current_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_rank3("current_tokens", current_tokens)
        batch_size, _num_tokens, teacher_dim = current_tokens.shape
        if teacher_dim != self.config.teacher_dim:
            raise ValueError(f"current_tokens last dim {teacher_dim} must match teacher_dim={self.config.teacher_dim}")
        _check_delta_tokens(delta_tokens, self.config, batch_size)
        if current_tokens.device != delta_tokens.device:
            raise ValueError(
                f"current_tokens device {current_tokens.device} must match delta_tokens device {delta_tokens.device}"
            )

        decoder_current = current_tokens if self.mode != "transition_only" else torch.zeros_like(current_tokens)
        decoder_delta = delta_tokens if self.mode != "current_only" else torch.zeros_like(delta_tokens)
        return self.decoder(
            decoder_current,
            decoder_delta,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )


class MatchedDeltaTokenEncoderProbe(nn.Module):
    """Predict frozen teacher tokens from current, future, or paired features.

    Inputs:
    - current_tokens: [B, N, D_teacher]
    - future_tokens: [B, N, D_teacher]
    - token_mask: optional [B, N]
    - horizon_ids: optional [B]

    Output:
    - pred_delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(self, config: DeltaTokenizerConfig, mode: TokenProbeMode) -> None:
        super().__init__()
        if mode not in TOKEN_PROBE_MODES:
            raise ValueError(f"mode must be one of {TOKEN_PROBE_MODES}, got {mode!r}")
        self.config = config
        self.mode = mode
        self.encoder = DeltaEncoder(config)

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_feature_pair(current_tokens, future_tokens)
        if current_tokens.shape[-1] != self.config.teacher_dim:
            raise ValueError(
                f"feature last dim {current_tokens.shape[-1]} must match teacher_dim={self.config.teacher_dim}"
            )

        encoder_current = current_tokens if self.mode != "future_only" else torch.zeros_like(current_tokens)
        encoder_future = future_tokens if self.mode != "current_only" else torch.zeros_like(future_tokens)
        return self.encoder(
            encoder_current,
            encoder_future,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )


def delta_token_loss_per_sample(
    pred_delta_tokens: torch.Tensor,
    target_delta_tokens: torch.Tensor,
    *,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Return matched token loss for each sample as a tensor shaped [B]."""

    _check_rank3("pred_delta_tokens", pred_delta_tokens)
    _check_rank3("target_delta_tokens", target_delta_tokens)
    if pred_delta_tokens.shape != target_delta_tokens.shape:
        raise ValueError(
            "pred_delta_tokens and target_delta_tokens must have identical shape, "
            f"got {tuple(pred_delta_tokens.shape)} and {tuple(target_delta_tokens.shape)}"
        )
    if pred_delta_tokens.device != target_delta_tokens.device:
        raise ValueError(
            f"pred_delta_tokens device {pred_delta_tokens.device} must match target device {target_delta_tokens.device}"
        )

    value = F.smooth_l1_loss(
        pred_delta_tokens.float(),
        target_delta_tokens.detach().float(),
        reduction="none",
    ).mean(dim=(1, 2))
    pred_norm = F.normalize(pred_delta_tokens.float(), dim=-1)
    target_norm = F.normalize(target_delta_tokens.detach().float(), dim=-1)
    cosine = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean(dim=1)
    return float(smooth_l1_weight) * value + float(cosine_weight) * cosine


def masked_geometry_loss_per_sample(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    token_mask: torch.Tensor | None,
    *,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Return masked feature reconstruction loss for each sample as [B]."""

    _check_feature_pair(pred_tokens, target_tokens)
    batch_size, num_tokens, _dim = pred_tokens.shape
    if token_mask is None:
        mask = torch.ones(batch_size, num_tokens, dtype=torch.float32, device=pred_tokens.device)
    else:
        if token_mask.shape != (batch_size, num_tokens):
            raise ValueError(f"token_mask must have shape {(batch_size, num_tokens)}, got {tuple(token_mask.shape)}")
        if token_mask.device != pred_tokens.device:
            raise ValueError(f"token_mask device {token_mask.device} must match token device {pred_tokens.device}")
        mask = token_mask.to(dtype=torch.float32)

    value = F.smooth_l1_loss(pred_tokens.float(), target_tokens.detach().float(), reduction="none").mean(dim=-1)
    pred_norm = F.normalize(pred_tokens.float(), dim=-1)
    target_norm = F.normalize(target_tokens.detach().float(), dim=-1)
    cosine = 1.0 - (pred_norm * target_norm).sum(dim=-1)
    per_token = float(smooth_l1_weight) * value + float(cosine_weight) * cosine
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (per_token * mask).sum(dim=1) / denominator


def build_swap_indices(
    primary_ids: torch.Tensor,
    *,
    relation: SwapRelation,
    seed: int,
    secondary_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic replacement indices and a per-row validity mask.

    Shapes:
    - primary_ids: [S]
    - secondary_ids: optional [S]
    - returned indices: [S]
    - returned valid: [S]
    """

    if relation not in SWAP_RELATIONS:
        raise ValueError(f"relation must be one of {SWAP_RELATIONS}, got {relation!r}")
    if primary_ids.ndim != 1:
        raise ValueError(f"primary_ids must have shape [S], got {tuple(primary_ids.shape)}")
    if secondary_ids is not None:
        if secondary_ids.shape != primary_ids.shape:
            raise ValueError(f"secondary_ids shape {tuple(secondary_ids.shape)} must match {tuple(primary_ids.shape)}")
        if secondary_ids.device != primary_ids.device:
            raise ValueError(
                f"secondary_ids device {secondary_ids.device} must match primary_ids device {primary_ids.device}"
            )
    if relation == "same_primary_different_secondary" and secondary_ids is None:
        raise ValueError("same_primary_different_secondary requires secondary_ids")

    source_device = primary_ids.device
    primary = primary_ids.detach().to(device="cpu")
    secondary = None if secondary_ids is None else secondary_ids.detach().to(device="cpu")
    sample_count = int(primary.shape[0])
    indices = torch.arange(sample_count, dtype=torch.long)
    valid = torch.zeros(sample_count, dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    all_rows = torch.arange(sample_count, dtype=torch.long)

    for row in range(sample_count):
        candidate = all_rows != row
        row_primary = primary[row].expand_as(primary)
        if relation == "same_primary":
            candidate &= primary == row_primary
        elif relation == "different_primary":
            candidate &= primary != row_primary
        else:
            if secondary is None:
                raise RuntimeError("secondary_ids unexpectedly missing")
            row_secondary = secondary[row].expand_as(secondary)
            candidate &= primary == row_primary
            candidate &= secondary != row_secondary

        candidate_rows = all_rows[candidate]
        if candidate_rows.numel() == 0:
            continue
        choice = torch.randint(int(candidate_rows.numel()), (1,), generator=generator).item()
        indices[row] = candidate_rows[int(choice)]
        valid[row] = True

    return indices.to(device=source_device), valid.to(device=source_device)
