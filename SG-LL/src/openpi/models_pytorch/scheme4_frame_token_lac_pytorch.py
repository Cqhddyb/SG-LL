"""Scheme4 policy variant using a matched future-state token target.

This file is intentionally separate from the accepted Route A implementation.
It keeps the same spatial branch, delta predictor, suffix fusion, loss weights,
and teacher-free inference path, changing only the frozen tokenizer target:

    frame target: Z_f = E_f(G_t+h), G_hat_t+h = D_f(Z_f)

The inherited predictor still receives only deployable current-time inputs at
policy training/inference.  Future images are used only to form the frozen
training target.
"""

from __future__ import annotations

import os

import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.future_state_tokenizer import FutureStateTokenizer
from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme4LACDoubleAlignPI0Pytorch


class Scheme4FrameTokenLACDoubleAlignPI0Pytorch(Scheme4LACDoubleAlignPI0Pytorch):
    """Matched future-state-token policy ablation for Scheme4 Route A."""

    def _load_scheme4_tokenizer(self, checkpoint_path: str | None) -> FutureStateTokenizer:
        if checkpoint_path is None:
            raise ValueError("scheme4_frame_token_lac_double_align requires extra_config.scheme4_tokenizer_checkpoint")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FutureStateTokenizer checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
            raise ValueError(f"Invalid FutureStateTokenizer checkpoint payload at {checkpoint_path}")
        schema_version = str(payload.get("schema_version", ""))
        if not schema_version.startswith("scheme4_frame_tokenizer_"):
            raise ValueError(
                "scheme4_frame_token_lac_double_align requires a FutureStateTokenizer checkpoint, "
                f"got schema_version={schema_version!r} at {checkpoint_path}"
            )
        tokenizer = FutureStateTokenizer(DeltaTokenizerConfig(**payload["config"]))
        tokenizer.load_state_dict(payload["state_dict"])
        tokenizer.eval()
        return tokenizer

    def _scheme4_teacher_delta_targets(
        self,
        teacher_current_hidden: torch.Tensor,
        teacher_future_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build frame-token targets without reading ``teacher_current_hidden``."""

        del teacher_current_hidden
        horizon_ids = self._scheme4_horizon_ids(teacher_future_hidden.shape[0], teacher_future_hidden.device)
        with torch.no_grad():
            target_frame_tokens = self.scheme4_tokenizer.encode(
                teacher_future_hidden.detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
            oracle_future_recon = self.scheme4_tokenizer.decode(
                target_frame_tokens.detach(),
                num_tokens=teacher_future_hidden.shape[1],
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
        return target_frame_tokens.detach(), oracle_future_recon.detach()

    def _scheme4_decode_future(
        self,
        spatial_tokens: torch.Tensor,
        pred_delta_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode predicted frame tokens without a current spatial reference."""

        horizon_ids = self._scheme4_horizon_ids(spatial_tokens.shape[0], spatial_tokens.device)
        return self.scheme4_tokenizer.decode(
            pred_delta_tokens,
            num_tokens=spatial_tokens.shape[1],
            token_mask=vision_mask,
            horizon_ids=horizon_ids,
        )
