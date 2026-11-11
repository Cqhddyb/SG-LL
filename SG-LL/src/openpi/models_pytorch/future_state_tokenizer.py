"""Matched future-state tokenizer for the transition-token semantics audit.

This module is deliberately independent from :mod:`delta_tokenizer`'s
``DeltaTokenizer`` wrapper.  It uses the same encoder and decoder building
blocks and the same ``DeltaTokenizerConfig`` capacity, but it never receives a
sample-specific current representation:

    Z_f = E_f(S_t+h)
    S_hat_t+h = D_f(Z_f)

Internally, the reused paired encoder receives a fixed all-zero current slot
and the reused residual decoder receives an all-zero current reference.  Those
slots preserve parameter-count and architectural parity with the transition
tokenizer; they do not contain information from the real ``S_t``.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from openpi.models_pytorch.delta_tokenizer import DeltaDecoder
from openpi.models_pytorch.delta_tokenizer import DeltaEncoder
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import geometry_distance


@dataclasses.dataclass(frozen=True)
class FutureStateTokenizerOutput:
    """Forward output for a future-state tokenizer.

    Shapes:
    - frame_tokens: [B, M_frame, D_frame]
    - future_recon: [B, N_teacher, D_teacher]
    """

    frame_tokens: torch.Tensor
    future_recon: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FutureStateTokenizerLossOutput:
    """Loss output for standalone frame-tokenizer training."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    model_output: FutureStateTokenizerOutput


def _check_future_tokens(future_tokens: torch.Tensor, config: DeltaTokenizerConfig) -> tuple[int, int]:
    if future_tokens.ndim != 3:
        raise ValueError(f"future_tokens must have shape [B, N, D], got {tuple(future_tokens.shape)}")
    batch_size, num_tokens, teacher_dim = future_tokens.shape
    if teacher_dim != config.teacher_dim:
        raise ValueError(f"future_tokens last dim {teacher_dim} must match config.teacher_dim={config.teacher_dim}")
    if num_tokens <= 0:
        raise ValueError("future_tokens must contain at least one token")
    if num_tokens > config.max_tokens:
        raise ValueError(f"future token count {num_tokens} exceeds config.max_tokens={config.max_tokens}")
    return int(batch_size), int(num_tokens)


def _check_frame_tokens(frame_tokens: torch.Tensor, config: DeltaTokenizerConfig) -> int:
    if frame_tokens.ndim != 3:
        raise ValueError(f"frame_tokens must have shape [B, M, D], got {tuple(frame_tokens.shape)}")
    batch_size, token_count, token_dim = frame_tokens.shape
    expected = (config.num_delta_tokens, config.delta_dim)
    if (token_count, token_dim) != expected:
        raise ValueError(
            f"frame_tokens must have shape [B, {expected[0]}, {expected[1]}], got {tuple(frame_tokens.shape)}"
        )
    return int(batch_size)


class FutureStateTokenizer(nn.Module):
    """Encode and decode compact future-state tokens with matched capacity.

    ``DeltaEncoder`` and ``DeltaDecoder`` are reused strictly for architecture
    parity.  ``encode`` constructs ``zeros_like(future_tokens)`` internally,
    and ``decode`` constructs a zero current reference from explicit batch and
    token dimensions.  Neither method accepts real current tokens.
    """

    def __init__(self, config: DeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DeltaEncoder(config)
        self.decoder = DeltaDecoder(config)

    def encode(
        self,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``future_tokens: [B,N,D_teacher]`` into ``[B,M,D_frame]``."""

        _check_future_tokens(future_tokens, self.config)
        fixed_zero_current = torch.zeros_like(future_tokens)
        return self.encoder(
            fixed_zero_current,
            future_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def decode(
        self,
        frame_tokens: torch.Tensor,
        *,
        num_tokens: int,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a future representation without a sample-specific current state.

        Args:
            frame_tokens: ``[B, M_frame, D_frame]``.
            num_tokens: target teacher-token count ``N``.  It is explicit so
                the method cannot obtain shape or content from ``S_t``.

        Returns:
            ``future_recon`` with shape ``[B, N, D_teacher]``.
        """

        batch_size = _check_frame_tokens(frame_tokens, self.config)
        if int(num_tokens) <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if int(num_tokens) > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds config.max_tokens={self.config.max_tokens}")
        fixed_zero_current = torch.zeros(
            (batch_size, int(num_tokens), self.config.teacher_dim),
            dtype=frame_tokens.dtype,
            device=frame_tokens.device,
        )
        return self.decoder(
            fixed_zero_current,
            frame_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def forward(
        self,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> FutureStateTokenizerOutput:
        """Return frame tokens and their reconstructed future representation."""

        _batch_size, num_tokens = _check_future_tokens(future_tokens, self.config)
        frame_tokens = self.encode(
            future_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        future_recon = self.decode(
            frame_tokens,
            num_tokens=num_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        return FutureStateTokenizerOutput(frame_tokens=frame_tokens, future_recon=future_recon)

    def config_dict(self) -> dict[str, object]:
        """Return the matched ``DeltaTokenizerConfig`` as serializable data."""

        return dataclasses.asdict(self.config)


def future_state_tokenizer_loss(
    tokenizer: FutureStateTokenizer,
    future_tokens: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    horizon_ids: torch.Tensor | None = None,
    camera_ids: torch.Tensor | None = None,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    relation_weight: float = 0.0,
) -> FutureStateTokenizerLossOutput:
    """Compute stop-gradient future reconstruction loss for the frame baseline."""

    output = tokenizer(
        future_tokens,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        camera_ids=camera_ids,
    )
    reconstruction_loss, reconstruction_metrics = geometry_distance(
        output.future_recon,
        future_tokens.detach(),
        token_mask,
        smooth_l1_weight=float(smooth_l1_weight),
        cosine_weight=float(cosine_weight),
        relation_weight=float(relation_weight),
        relation_max_tokens=tokenizer.config.relation_max_tokens,
    )
    metrics = {
        "future_state_tokenizer_loss": reconstruction_loss.detach(),
        "future_state_rec_loss": reconstruction_loss.detach(),
        **{f"future_state_{key}": value.detach() for key, value in reconstruction_metrics.items()},
    }
    return FutureStateTokenizerLossOutput(
        loss=reconstruction_loss,
        metrics=metrics,
        model_output=output,
    )
