"""Utility-supervised selective fusion for the fixed-horizon Scheme4 path.

The module deliberately changes only the scale of the existing future-token
suffix residual.  A zero scale is an exact no-lookahead action path and leaves
the independent spatial-prefix pathway untouched.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

LOOKAHEAD_ROUTING_MODES = ("none", "fixed", "soft", "hard")


class Scheme4UtilityRouter(nn.Module):
    """Predict whether the fixed Scheme4 future residual should be used.

    Inputs are current policy-available features only:
    - spatial tokens ``[B, N, D_spatial]``
    - masked language summary ``[B, D_language]``
    - robot state ``[B, D_state]``
    - detached action intent ``[B, D_intent]``

    The output is a per-example probability in ``[0, 1]``.  It selects whether
    to fuse the existing configured horizon; it does not predict a new horizon.
    """

    def __init__(
        self,
        *,
        spatial_dim: int,
        language_dim: int,
        state_dim: int,
        action_intent_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        dims = {
            "spatial_dim": spatial_dim,
            "language_dim": language_dim,
            "state_dim": state_dim,
            "action_intent_dim": action_intent_dim,
            "hidden_dim": hidden_dim,
        }
        for name, value in dims.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.spatial_dim = int(spatial_dim)
        self.language_dim = int(language_dim)
        self.state_dim = int(state_dim)
        self.action_intent_dim = int(action_intent_dim)
        self.hidden_dim = int(hidden_dim)

        self.spatial_encoder = self._feature_encoder(self.spatial_dim, self.hidden_dim)
        self.language_encoder = self._feature_encoder(self.language_dim, self.hidden_dim)
        self.state_encoder = self._feature_encoder(self.state_dim, self.hidden_dim)
        self.action_intent_encoder = self._feature_encoder(self.action_intent_dim, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _feature_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must have shape [B, N, D], got {tuple(spatial_tokens.shape)}")
        batch_size, num_tokens, spatial_dim = spatial_tokens.shape
        if spatial_dim != self.spatial_dim:
            raise ValueError(f"spatial_tokens last dim must be {self.spatial_dim}, got {spatial_dim}")
        if token_mask is not None and token_mask.shape != (batch_size, num_tokens):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} must match {(batch_size, num_tokens)}")
        expected = {
            "language_features": (batch_size, self.language_dim),
            "state": (batch_size, self.state_dim),
            "action_intent": (batch_size, self.action_intent_dim),
        }
        actual = {
            "language_features": tuple(language_features.shape),
            "state": tuple(state.shape),
            "action_intent": tuple(action_intent.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} shape {actual[name]} must match {shape}")

        spatial = spatial_tokens.float()
        if token_mask is None:
            spatial_summary = spatial.mean(dim=1)
        else:
            mask = token_mask.to(device=spatial.device, dtype=spatial.dtype).unsqueeze(-1)
            spatial_summary = (spatial * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        encoded = torch.cat(
            (
                self.spatial_encoder(spatial_summary),
                self.language_encoder(language_features.float()),
                self.state_encoder(state.float()),
                self.action_intent_encoder(action_intent.float()),
            ),
            dim=-1,
        )
        return torch.sigmoid(self.fusion(encoded).squeeze(-1))


def resolve_lookahead_scale(probability: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert per-example router probabilities into bounded fusion scales."""

    if probability.ndim != 1:
        raise ValueError(f"probability must have shape [B], got {tuple(probability.shape)}")
    if not torch.isfinite(probability).all():
        raise ValueError("probability must contain only finite values")
    if torch.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probability values must be in [0, 1]")
    if mode not in LOOKAHEAD_ROUTING_MODES:
        raise ValueError(f"mode must be one of {LOOKAHEAD_ROUTING_MODES}, got {mode!r}")
    if mode == "none":
        return torch.zeros_like(probability)
    if mode == "fixed":
        return torch.ones_like(probability)
    if mode == "hard":
        return (probability >= 0.5).to(dtype=probability.dtype)
    return probability


def training_fusion_scale(probability: torch.Tensor, *, router_only: bool) -> torch.Tensor:
    """Keep router-only optimization on its declared utility objective."""
    if probability.ndim != 1 or not torch.isfinite(probability).all():
        raise ValueError("probability must be a finite tensor with shape [B]")
    return probability.detach() if router_only else probability


def utility_targets_from_losses(
    base_losses: torch.Tensor,
    lookahead_losses: torch.Tensor,
    *,
    temperature: float,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached action-loss benefit and its soft binary utility target.

    ``benefit = base_loss - lookahead_loss`` is an offline action-loss proxy,
    not a task-success label.  Ground-truth actions appear only in these
    training losses and are never passed to the router.
    """

    if base_losses.ndim != 1 or lookahead_losses.ndim != 1:
        raise ValueError("base_losses and lookahead_losses must have shape [B]")
    if base_losses.shape != lookahead_losses.shape:
        raise ValueError(
            f"base_losses shape {tuple(base_losses.shape)} must match lookahead_losses {tuple(lookahead_losses.shape)}"
        )
    if not torch.isfinite(base_losses).all() or not torch.isfinite(lookahead_losses).all():
        raise ValueError("action losses must contain only finite values")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not torch.isfinite(torch.tensor(float(margin))):
        raise ValueError(f"margin must be finite, got {margin}")

    benefit = (base_losses.detach().float() - lookahead_losses.detach().float()).detach()
    target = torch.sigmoid((benefit - float(margin)) / float(temperature)).detach()
    return benefit, target


def utility_supervision_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for a soft action-loss utility target."""

    if probability.ndim != 1 or target.ndim != 1:
        raise ValueError("probability and target must have shape [B]")
    if probability.shape != target.shape:
        raise ValueError(f"probability shape {tuple(probability.shape)} must match target {tuple(target.shape)}")
    if not torch.isfinite(probability).all() or not torch.isfinite(target).all():
        raise ValueError("probability and target must contain only finite values")
    eps = torch.finfo(probability.dtype).eps
    return F.binary_cross_entropy(probability.clamp(min=eps, max=1.0 - eps), target.to(probability.dtype))
