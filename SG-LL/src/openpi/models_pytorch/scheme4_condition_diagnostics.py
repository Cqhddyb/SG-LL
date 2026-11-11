"""Condition-use diagnostics for Scheme4 delta prediction.

These helpers are policy-side diagnostics.  They do not instantiate pi0.5 and
do not change inference behavior by themselves.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def roll_batch_condition(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically mismatch a batch condition by rolling rows.

    Shape contract:
    - value: [B, ...]
    - output: [B, ...]
    - valid: scalar bool tensor indicating whether B > 1
    """

    if value.ndim < 1:
        raise ValueError(f"value must have a batch dimension, got shape {tuple(value.shape)}")
    valid = torch.tensor(value.shape[0] > 1, dtype=torch.bool, device=value.device)
    if value.shape[0] <= 1:
        return value.clone(), valid
    return torch.roll(value, shifts=1, dims=0), valid


def delta_token_distance(pred_delta_tokens: torch.Tensor, target_delta_tokens: torch.Tensor) -> torch.Tensor:
    """Token distance used for Scheme4 negative-condition diagnostics.

    Shape contract:
    - pred_delta_tokens: [B, M_delta, D_delta]
    - target_delta_tokens: [B, M_delta, D_delta]
    """

    if pred_delta_tokens.shape != target_delta_tokens.shape:
        raise ValueError(
            "pred_delta_tokens and target_delta_tokens must have identical shape, got "
            f"{tuple(pred_delta_tokens.shape)} and {tuple(target_delta_tokens.shape)}"
        )
    if pred_delta_tokens.ndim != 3:
        raise ValueError(f"delta tokens must be rank 3, got shape {tuple(pred_delta_tokens.shape)}")
    smooth_l1 = F.smooth_l1_loss(pred_delta_tokens.float(), target_delta_tokens.float())
    pred_norm = F.normalize(pred_delta_tokens.float(), dim=-1)
    target_norm = F.normalize(target_delta_tokens.float(), dim=-1)
    cosine = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
    return smooth_l1 + 0.1 * cosine


def margin_negative_condition_loss(
    normal_distance: torch.Tensor,
    negative_distance: torch.Tensor,
    *,
    margin: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Encourage normal-condition predictions to beat negative conditions.

    The loss is zero when the batch cannot be mismatched, e.g. B=1.
    """

    if normal_distance.ndim != 0 or negative_distance.ndim != 0:
        raise ValueError("normal_distance and negative_distance must be scalar tensors")
    if valid.ndim != 0:
        raise ValueError(f"valid must be a scalar tensor, got shape {tuple(valid.shape)}")
    raw = F.relu(
        torch.as_tensor(float(margin), dtype=normal_distance.dtype, device=normal_distance.device)
        + normal_distance
        - negative_distance
    )
    return torch.where(valid.to(device=raw.device), raw, torch.zeros_like(raw))
