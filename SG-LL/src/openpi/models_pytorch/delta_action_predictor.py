"""Auxiliary action-conditioned delta predictor for scheme3.

This module is intentionally standalone.  It predicts compact decodable delta
tokens from current teacher tokens plus an action/state condition, but it does
not import or modify any pi0 policy classes.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class DeltaActionPredictorConfig:
    """Configuration for the auxiliary GT-action delta predictor."""

    teacher_dim: int = 2048
    delta_dim: int = 256
    num_delta_tokens: int = 8
    action_horizon: int = 10
    action_dim: int = 32
    state_dim: int = 32
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0
    max_horizon: int = 64
    zero_init_output: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "teacher_dim": self.teacher_dim,
            "delta_dim": self.delta_dim,
            "num_delta_tokens": self.num_delta_tokens,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "max_horizon": self.max_horizon,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if float(self.dropout) < 0.0:
            raise ValueError(f"dropout must be non-negative, got {self.dropout}")


@dataclasses.dataclass(frozen=True)
class DeltaActionPredictorLossOutput:
    """Loss output for auxiliary delta prediction.

    Shapes:
    - pred_delta_tokens: [B, M_delta, D_delta]
    """

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    pred_delta_tokens: torch.Tensor


def _assert_rank3(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must be rank 3, got shape {tuple(value.shape)}")


def _check_current_tokens(current_tokens: torch.Tensor, *, teacher_dim: int) -> tuple[int, int]:
    _assert_rank3("current_tokens", current_tokens)
    batch_size, num_tokens, dim = current_tokens.shape
    if dim != teacher_dim:
        raise ValueError(f"current_tokens last dim {dim} does not match teacher_dim={teacher_dim}")
    return int(batch_size), int(num_tokens)


def _check_actions(actions: torch.Tensor, *, batch_size: int, action_horizon: int, action_dim: int) -> None:
    expected = (batch_size, action_horizon, action_dim)
    if actions.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")


def _check_state(state: torch.Tensor | None, *, batch_size: int, state_dim: int, device: torch.device) -> torch.Tensor:
    if state is None:
        return torch.zeros(batch_size, state_dim, device=device, dtype=torch.float32)
    expected = (batch_size, state_dim)
    if state.shape != expected:
        raise ValueError(f"state must have shape {expected}, got {tuple(state.shape)}")
    if state.device != device:
        raise ValueError(f"state device {state.device} does not match current_tokens device {device}")
    return state


def _check_token_mask(
    token_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if token_mask is None:
        return None
    expected = (batch_size, num_tokens)
    if token_mask.shape != expected:
        raise ValueError(f"token_mask must have shape {expected}, got {tuple(token_mask.shape)}")
    if token_mask.device != device:
        raise ValueError(f"token_mask device {token_mask.device} does not match current_tokens device {device}")
    return token_mask.bool()


def _check_horizon_ids(
    horizon_ids: torch.Tensor | None,
    *,
    batch_size: int,
    max_horizon: int,
    device: torch.device,
) -> torch.Tensor | None:
    if horizon_ids is None:
        return None
    expected = (batch_size,)
    if horizon_ids.shape != expected:
        raise ValueError(f"horizon_ids must have shape {expected}, got {tuple(horizon_ids.shape)}")
    if horizon_ids.device != device:
        raise ValueError(f"horizon_ids device {horizon_ids.device} does not match current_tokens device {device}")
    if torch.is_floating_point(horizon_ids) or torch.is_complex(horizon_ids):
        raise ValueError(f"horizon_ids must be an integer tensor, got dtype={horizon_ids.dtype}")
    ids = horizon_ids.long()
    if ids.numel() > 0 and (int(ids.min()) < 0 or int(ids.max()) >= max_horizon):
        raise ValueError(f"horizon_ids must be in [0, {max_horizon}), got min={int(ids.min())}, max={int(ids.max())}")
    return ids


def _masked_mean(tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return tokens.float().mean(dim=1)
    mask = token_mask.to(dtype=torch.float32).unsqueeze(-1)
    return (tokens.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int,
    dropout: float,
    input_layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    if input_layer_norm:
        layers.append(nn.LayerNorm(input_dim))
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
    for _ in range(max(0, num_layers - 2)):
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
    layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)])
    return nn.Sequential(*layers)


class DeltaActionPredictor(nn.Module):
    """Predict decodable delta tokens from current teacher tokens and action condition.

    Input shapes:
    - current_tokens: [B, N_teacher, D_teacher]
    - actions: [B, H_action, A_dim]
    - state: optional [B, D_state]
    - token_mask: optional [B, N_teacher]
    - horizon_ids: optional [B]

    Output shape:
    - pred_delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(self, config: DeltaActionPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.current_encoder = _make_mlp(
            config.teacher_dim,
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            input_layer_norm=True,
        )
        self.action_encoder = _make_mlp(
            config.action_horizon * config.action_dim,
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            input_layer_norm=True,
        )
        self.state_encoder = _make_mlp(
            config.state_dim,
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            input_layer_norm=True,
        )
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.fusion = _make_mlp(
            config.hidden_dim * 4,
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            input_layer_norm=True,
        )
        self.delta_queries = nn.Parameter(
            torch.randn(config.num_delta_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.delta_head = _make_mlp(
            config.hidden_dim,
            config.hidden_dim,
            config.delta_dim,
            num_layers=2,
            dropout=config.dropout,
            input_layer_norm=True,
        )
        if config.zero_init_output:
            final = self.delta_head[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("delta_head final layer is expected to be nn.Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _horizon_features(
        self,
        horizon_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        ids = _check_horizon_ids(
            horizon_ids,
            batch_size=batch_size,
            max_horizon=self.config.max_horizon,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, self.config.hidden_dim, dtype=torch.float32, device=device)
        return self.horizon_embed(ids).float()

    def forward(
        self,
        current_tokens: torch.Tensor,
        actions: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_tokens = _check_current_tokens(current_tokens, teacher_dim=self.config.teacher_dim)
        device = current_tokens.device
        if actions.device != device:
            raise ValueError(f"actions device {actions.device} does not match current_tokens device {device}")
        _check_actions(
            actions,
            batch_size=batch_size,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
        )
        state = _check_state(state, batch_size=batch_size, state_dim=self.config.state_dim, device=device)
        token_mask = _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=device)
        horizon = self._horizon_features(horizon_ids, batch_size=batch_size, device=device)

        current_summary = _masked_mean(current_tokens, token_mask)
        current_features = self.current_encoder(current_summary.float())
        action_features = self.action_encoder(actions.float().reshape(batch_size, -1))
        state_features = self.state_encoder(state.float())
        fused = self.fusion(torch.cat([current_features, action_features, state_features, horizon], dim=-1))

        queries = fused.unsqueeze(1) + self.delta_queries.to(device=device, dtype=fused.dtype).unsqueeze(0)
        pred_delta = self.delta_head(queries)
        expected = (batch_size, self.config.num_delta_tokens, self.config.delta_dim)
        if pred_delta.shape != expected:
            raise RuntimeError(f"pred_delta must have shape {expected}, got {tuple(pred_delta.shape)}")
        return pred_delta

    def config_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self.config)


def _check_target_delta(
    target_delta_tokens: torch.Tensor,
    *,
    batch_size: int,
    num_delta_tokens: int,
    delta_dim: int,
    device: torch.device,
) -> None:
    expected = (batch_size, num_delta_tokens, delta_dim)
    if target_delta_tokens.shape != expected:
        raise ValueError(f"target_delta_tokens must have shape {expected}, got {tuple(target_delta_tokens.shape)}")
    if target_delta_tokens.device != device:
        raise ValueError(
            f"target_delta_tokens device {target_delta_tokens.device} does not match current_tokens device {device}"
        )


def delta_action_predictor_loss(
    predictor: DeltaActionPredictor,
    current_tokens: torch.Tensor,
    actions: torch.Tensor,
    target_delta_tokens: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    horizon_ids: torch.Tensor | None = None,
    smooth_l1_weight: float = 1.0,
    mse_weight: float = 0.0,
    cosine_weight: float = 0.1,
) -> DeltaActionPredictorLossOutput:
    """Compute auxiliary delta-token prediction loss.

    The target delta tokens are expected to be detached by the caller when they
    come from a frozen tokenizer.
    """

    pred_delta = predictor(
        current_tokens,
        actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
    )
    batch_size = int(current_tokens.shape[0])
    _check_target_delta(
        target_delta_tokens,
        batch_size=batch_size,
        num_delta_tokens=predictor.config.num_delta_tokens,
        delta_dim=predictor.config.delta_dim,
        device=current_tokens.device,
    )

    pred = pred_delta.float()
    target = target_delta_tokens.float()
    smooth_l1_loss = F.smooth_l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    cosine_loss = 1.0 - (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()
    l1_loss = (pred - target).abs().mean()
    loss = float(smooth_l1_weight) * smooth_l1_loss + float(mse_weight) * mse_loss + float(cosine_weight) * cosine_loss
    metrics = {
        "delta_action_loss": loss.detach(),
        "delta_smooth_l1_loss": smooth_l1_loss.detach(),
        "delta_mse_loss": mse_loss.detach(),
        "delta_cosine_loss": cosine_loss.detach(),
        "delta_l1_loss": l1_loss.detach(),
    }
    return DeltaActionPredictorLossOutput(loss=loss, metrics=metrics, pred_delta_tokens=pred_delta)
