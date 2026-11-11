"""Proposal model variants for pi0.5 LIBERO geometry supervision.

This file intentionally does not replace ``pi0_align_pytorch.py``.  It contains
two experimental variants matching the two designs in the historical research notes.
Both variants keep VGGT out of inference and keep the PaliGemma/VLM backbone
freezable through the existing training script policy.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictor
from openpi.models_pytorch.delta_action_predictor import DeltaActionPredictorConfig
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizer
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import geometry_distance
from openpi.models_pytorch.flow_matching_utils import estimate_action_endpoint
import openpi.models_pytorch.geo_action_modules as _geo_modules
from openpi.models_pytorch.pi0_align_pytorch import PI0Pytorch as GeometryResidualPI0Pytorch
from openpi.models_pytorch.pi0_align_pytorch import make_att_2d_masks
from openpi.models_pytorch.scheme4_adaptive_lookahead import LOOKAHEAD_ROUTING_MODES
from openpi.models_pytorch.scheme4_adaptive_lookahead import Scheme4UtilityRouter
from openpi.models_pytorch.scheme4_adaptive_lookahead import resolve_lookahead_scale
from openpi.models_pytorch.scheme4_adaptive_lookahead import training_fusion_scale
from openpi.models_pytorch.scheme4_adaptive_lookahead import utility_supervision_loss
from openpi.models_pytorch.scheme4_adaptive_lookahead import utility_targets_from_losses
from openpi.models_pytorch.scheme4_condition_diagnostics import delta_token_distance
from openpi.models_pytorch.scheme4_condition_diagnostics import margin_negative_condition_loss
from openpi.models_pytorch.scheme4_condition_diagnostics import roll_batch_condition
from openpi.models_pytorch.scheme4_inference_diagnostics import finalize_request_diagnostics
from openpi.models_pytorch.scheme4_inference_diagnostics import flow_step_diagnostics


def select_scheme4_training_intent_suffix(
    sampled_flow_suffix: torch.Tensor,
    initial_noise_suffix: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Select the policy-available suffix used as the future-predictor intent."""
    if sampled_flow_suffix.ndim != 3:
        raise ValueError(f"sampled_flow_suffix must be [B,H,D], got {tuple(sampled_flow_suffix.shape)}")
    if mode == "sampled_flow":
        return sampled_flow_suffix
    if mode != "t1_noise":
        raise ValueError(f"training intent timestep must be 'sampled_flow' or 't1_noise', got {mode!r}")
    if initial_noise_suffix is None:
        raise ValueError("initial_noise_suffix is required for t1_noise training intent")
    if initial_noise_suffix.shape != sampled_flow_suffix.shape:
        raise ValueError(
            "initial_noise_suffix shape must match sampled_flow_suffix, got "
            f"{tuple(initial_noise_suffix.shape)} vs {tuple(sampled_flow_suffix.shape)}"
        )
    return initial_noise_suffix


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _use_eager_sample_actions(module: nn.Module) -> None:
    """Proposal modules are easier to validate in eager mode than max-autotune."""
    if os.environ.get("OPENPI_SF_PROPOSAL_COMPILE", "0").lower() in ("1", "true", "yes", "y"):
        return
    module.sample_actions = type(module).sample_actions.__get__(module, type(module))


class ThreeLayerResidualMLP(nn.Module):
    """Token-wise 3-layer MLP with a residual connection."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


def _masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = F.layer_norm(pred.float(), pred.shape[-1:])
    target = F.layer_norm(target.float(), target.shape[-1:])
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.bool()
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def _masked_relation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    max_tokens: int = 128,
) -> torch.Tensor:
    if pred.shape[1] > max_tokens:
        idx = torch.linspace(0, pred.shape[1] - 1, steps=max_tokens, device=pred.device).round().long()
        pred = pred.index_select(dim=1, index=idx)
        target = target.index_select(dim=1, index=idx)
        if mask is not None:
            mask = mask.index_select(dim=1, index=idx)

    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    pred_sim = torch.matmul(pred, pred.transpose(1, 2))
    target_sim = torch.matmul(target, target.transpose(1, 2))
    loss = (pred_sim - target_sim).abs()
    if mask is None:
        return loss.mean()
    mask = mask.bool()
    pair_mask = mask[:, :, None] & mask[:, None, :]
    denom = pair_mask.float().sum().clamp_min(1.0)
    return (loss * pair_mask.float()).sum() / denom


def _spatial_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    value_coeff: float = 0.1,
    relation_coeff: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_loss = _geo_modules.masked_cosine_loss(pred, target, mask)
    value_loss = _masked_smooth_l1_loss(pred, target, mask)
    relation_loss = _masked_relation_loss(pred, target, mask)
    loss = token_loss + float(value_coeff) * value_loss + float(relation_coeff) * relation_loss
    return loss, {
        "spatial_token_cos_loss": token_loss.detach(),
        "spatial_value_loss": value_loss.detach(),
        "spatial_relation_loss": relation_loss.detach(),
    }


class SwiGLUResidualBlock(nn.Module):
    """Pre-LN SwiGLU block with a zero-init residual branch."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x.float())
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        return x.float() + self.down_proj(self.dropout(h))


class SwiGLUResidualStack(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([SwiGLUResidualBlock(dim, hidden_dim, dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        return x


class SpatialQueryResampler(nn.Module):
    """Perceiver-style learnable spatial queries over visual tokens."""

    def __init__(self, input_dim: int, query_dim: int, num_queries: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) / math.sqrt(query_dim))
        self.input_proj = nn.Identity() if input_dim == query_dim else nn.Linear(input_dim, query_dim)
        self.cross_attn = _geo_modules.CrossAttentionBlock(query_dim, num_heads, dropout)
        self.out_norm = nn.LayerNorm(query_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.input_proj(hidden.float())
        queries = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1).to(dtype=memory.dtype, device=memory.device)
        return self.out_norm(self.cross_attn(queries, memory, mask))


class SuffixQueryFusion(nn.Module):
    """Zero-gated residual update from spatial query tokens into suffix states."""

    def __init__(self, suffix_dim: int, query_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, suffix_dim)
        self.cross_attn = _geo_modules.CrossAttentionBlock(suffix_dim, num_heads, dropout)
        self.delta = nn.Sequential(
            nn.LayerNorm(suffix_dim),
            nn.Linear(suffix_dim, suffix_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(suffix_dim * 2, suffix_dim),
        )
        self.gate = nn.Parameter(torch.zeros(suffix_dim))

    def forward(self, suffix_out: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.query_proj(query_tokens.float())
        fused = self.cross_attn(suffix_out.float(), memory)
        gate = torch.tanh(self.gate).to(dtype=fused.dtype, device=fused.device).view(1, 1, -1)
        return suffix_out.float() + gate * self.delta(fused)


class Scheme1ResidualQueryAdapter(nn.Module):
    """Scheme1-RQ: residual visual adapter plus spatial queries for action fusion."""

    def __init__(
        self,
        visual_dim: int,
        suffix_dim: int,
        teacher_dim: int,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        max_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_adapter = SwiGLUResidualStack(visual_dim, hidden_dim, num_blocks=2, dropout=dropout)
        self.query_resampler = SpatialQueryResampler(visual_dim, visual_dim, num_queries, num_heads, dropout)
        self.teacher_decoder = _geo_modules.TeacherTokenDecoder(
            hidden_dim=visual_dim,
            target_dim=teacher_dim,
            max_tokens=max_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.suffix_fusion = SuffixQueryFusion(suffix_dim, visual_dim, num_heads, dropout)

    def encode_visual(
        self, vision_hidden: torch.Tensor, vision_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_visual = self.visual_adapter(vision_hidden.float())
        query_tokens = self.query_resampler(refined_visual, vision_mask)
        return refined_visual, query_tokens

    def predict_spatial(self, query_tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
        return self.teacher_decoder(query_tokens.float(), num_tokens)

    def refine_suffix(self, suffix_out: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        return self.suffix_fusion(suffix_out, query_tokens)


class Scheme2GatedDualHeadAdapter(nn.Module):
    """Scheme2-G: shared spatial trunk, separate alignment/action heads, zero-gated injection."""

    def __init__(
        self,
        visual_dim: int,
        teacher_dim: int,
        hidden_dim: int,
        max_aux_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_aux_tokens = int(max_aux_tokens)
        self.trunk = SwiGLUResidualStack(visual_dim, hidden_dim, num_blocks=2, dropout=dropout)
        self.align_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
        )
        self.action_norm = nn.LayerNorm(visual_dim)
        self.gate = nn.Parameter(torch.zeros(visual_dim))

    def _select_aux_tokens(
        self, aux_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.max_aux_tokens <= 0 or aux_tokens.shape[1] <= self.max_aux_tokens:
            return aux_tokens, mask
        idx = (
            torch.linspace(0, aux_tokens.shape[1] - 1, steps=self.max_aux_tokens, device=aux_tokens.device)
            .round()
            .long()
        )
        selected_aux = aux_tokens.index_select(dim=1, index=idx)
        selected_mask = None if mask is None else mask.index_select(dim=1, index=idx)
        return selected_aux, selected_mask

    def forward(
        self,
        vision_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        *,
        select_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(vision_hidden.float())
        pred_teacher = self.align_head(hidden)
        aux_tokens = self.action_norm(self.action_head(hidden))
        gate = torch.tanh(self.gate).to(dtype=aux_tokens.dtype, device=aux_tokens.device).view(1, 1, -1)
        aux_tokens = aux_tokens * gate
        if vision_mask is not None:
            aux_tokens = aux_tokens * vision_mask.to(dtype=aux_tokens.dtype).unsqueeze(-1)
        aux_tokens, aux_mask = (
            self._select_aux_tokens(aux_tokens, vision_mask) if select_aux else (aux_tokens, vision_mask)
        )
        return pred_teacher, aux_tokens, aux_mask


class Scheme1SharedMLPSpatialActionAdapter(nn.Module):
    """Shared MLP used by both action flow and VGGT spatial supervision.

    Visual tokens are refined in PaliGemma hidden width and decoded to VGGT
    tokens.  Suffix/action-expert tokens are projected into the same bridge
    width, passed through the same MLP, then projected back before the action
    output head.  Therefore the same MLP receives gradients from both action
    flow loss and spatial alignment loss.
    """

    def __init__(
        self,
        visual_dim: int,
        suffix_dim: int,
        teacher_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.shared_mlp = ThreeLayerResidualMLP(visual_dim, hidden_dim, dropout)
        self.suffix_to_visual = nn.Identity() if suffix_dim == visual_dim else nn.Linear(suffix_dim, visual_dim)
        self.visual_to_suffix_delta = nn.Linear(visual_dim, suffix_dim)
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )
        nn.init.zeros_(self.visual_to_suffix_delta.weight)
        nn.init.zeros_(self.visual_to_suffix_delta.bias)

    def refine_visual(self, vision_hidden: torch.Tensor) -> torch.Tensor:
        return self.shared_mlp(vision_hidden.float())

    def predict_spatial(self, refined_visual: torch.Tensor) -> torch.Tensor:
        return self.spatial_head(refined_visual.float())

    def refine_suffix(self, suffix_out: torch.Tensor) -> torch.Tensor:
        bridge_tokens = self.suffix_to_visual(suffix_out.float())
        refined_bridge = self.shared_mlp(bridge_tokens)
        refined_suffix = suffix_out.float() + self.visual_to_suffix_delta(refined_bridge)
        return refined_suffix.to(dtype=suffix_out.dtype, device=suffix_out.device)


class Scheme2VisualConcatAdapter(nn.Module):
    """Predict VGGT-aligned tokens and down-project them as action-prefix tokens."""

    def __init__(
        self,
        visual_dim: int,
        teacher_dim: int,
        hidden_dim: int,
        max_aux_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_aux_tokens = int(max_aux_tokens)
        self.trunk = ThreeLayerResidualMLP(visual_dim, hidden_dim, dropout)
        self.to_teacher = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )
        self.to_aux_prefix = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
        )
        nn.init.zeros_(self.to_aux_prefix[-1].weight)
        nn.init.zeros_(self.to_aux_prefix[-1].bias)

    def _select_aux_tokens(
        self, aux_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.max_aux_tokens <= 0 or aux_tokens.shape[1] <= self.max_aux_tokens:
            return aux_tokens, mask

        # Evenly spaced deterministic subsampling keeps token order stable for
        # causal/prefix position ids while bounding the extra prefix length.
        idx = (
            torch.linspace(
                0,
                aux_tokens.shape[1] - 1,
                steps=self.max_aux_tokens,
                device=aux_tokens.device,
            )
            .round()
            .long()
        )
        selected_aux = aux_tokens.index_select(dim=1, index=idx)
        selected_mask = None if mask is None else mask.index_select(dim=1, index=idx)
        return selected_aux, selected_mask

    def forward(
        self,
        vision_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        *,
        select_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(vision_hidden.float())
        pred_teacher = self.to_teacher(hidden)
        aux_tokens = self.to_aux_prefix(hidden)
        if vision_mask is not None:
            aux_tokens = aux_tokens * vision_mask.to(dtype=aux_tokens.dtype).unsqueeze(-1)
        aux_tokens, aux_mask = (
            self._select_aux_tokens(aux_tokens, vision_mask) if select_aux else (aux_tokens, vision_mask)
        )
        return pred_teacher, aux_tokens, aux_mask


class Scheme2UnifiedSpatialBottleneckAdapter(nn.Module):
    """Corrected Scheme2 adapter with one spatial bottleneck for loss and action.

    Shape contract:
    - vision_hidden: [B, N_visual, D_visual]
    - vision_mask: optional [B, N_visual]
    - spatial_tokens: [B, N_visual, D_spatial]
    - aux_tokens: [B, N_aux, D_visual]

    The same spatial_tokens tensor is supervised against VGGT current tokens and
    projected into the prefix residual path.  The residual gate is exactly zero
    at initialization, so the action path is identity until the gate learns.
    """

    def __init__(
        self,
        visual_dim: int,
        spatial_dim: int,
        hidden_dim: int,
        max_aux_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.spatial_dim = int(spatial_dim)
        self.max_aux_tokens = int(max_aux_tokens)
        self.to_spatial = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.spatial_dim),
        )
        self.to_action_residual = nn.Sequential(
            nn.LayerNorm(self.spatial_dim),
            nn.Linear(self.spatial_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.visual_dim),
        )
        self.gate = nn.Parameter(torch.zeros(self.visual_dim))

    def _select_aux_tokens(
        self, aux_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.max_aux_tokens <= 0 or aux_tokens.shape[1] <= self.max_aux_tokens:
            return aux_tokens, mask

        idx = (
            torch.linspace(0, aux_tokens.shape[1] - 1, steps=self.max_aux_tokens, device=aux_tokens.device)
            .round()
            .long()
        )
        selected_aux = aux_tokens.index_select(dim=1, index=idx)
        selected_mask = None if mask is None else mask.index_select(dim=1, index=idx)
        return selected_aux, selected_mask

    def forward(
        self,
        vision_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        *,
        select_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if vision_hidden.ndim != 3:
            raise ValueError(f"vision_hidden must be rank 3, got shape {tuple(vision_hidden.shape)}")
        if vision_hidden.shape[-1] != self.visual_dim:
            raise ValueError(
                f"vision_hidden last dim {vision_hidden.shape[-1]} does not match visual_dim={self.visual_dim}"
            )
        if vision_mask is not None and vision_mask.shape != vision_hidden.shape[:2]:
            raise ValueError(
                f"vision_mask shape {tuple(vision_mask.shape)} must match vision_hidden first dims "
                f"{tuple(vision_hidden.shape[:2])}"
            )

        spatial_tokens = self.to_spatial(vision_hidden.float())
        if spatial_tokens.shape != (vision_hidden.shape[0], vision_hidden.shape[1], self.spatial_dim):
            raise RuntimeError(
                f"spatial_tokens shape {tuple(spatial_tokens.shape)} does not match expected "
                f"{(vision_hidden.shape[0], vision_hidden.shape[1], self.spatial_dim)}"
            )

        action_residual = self.to_action_residual(spatial_tokens)
        if action_residual.shape != vision_hidden.shape:
            raise RuntimeError(
                f"action_residual shape {tuple(action_residual.shape)} does not match vision_hidden "
                f"{tuple(vision_hidden.shape)}"
            )
        gate = (
            torch.tanh(self.gate)
            .to(dtype=action_residual.dtype, device=action_residual.device)
            .view(1, 1, self.visual_dim)
        )
        aux_tokens = action_residual * gate
        if vision_mask is not None:
            aux_tokens = aux_tokens * vision_mask.to(dtype=aux_tokens.dtype, device=aux_tokens.device).unsqueeze(-1)

        aux_tokens, aux_mask = (
            self._select_aux_tokens(aux_tokens, vision_mask) if select_aux else (aux_tokens, vision_mask)
        )
        return spatial_tokens, aux_tokens, aux_mask


class DirectVGGTInjectionAdapter(nn.Module):
    """Project online VGGT current tokens into a zero-gated prefix residual.

    Shape contract:
    - teacher_tokens: [B, N_teacher, D_teacher]
    - token_mask: optional [B, N_teacher]
    - aux_tokens: [B, N_aux, D_visual] in concat mode or [B, N_teacher, D_visual]

    This adapter is intentionally a direct-online-teacher baseline: the input
    is already VGGT current geometry, so no spatial prediction loss is used.
    """

    def __init__(
        self,
        teacher_dim: int,
        visual_dim: int,
        hidden_dim: int,
        max_aux_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.visual_dim = int(visual_dim)
        self.max_aux_tokens = int(max_aux_tokens)
        self.to_action_residual = nn.Sequential(
            nn.LayerNorm(self.teacher_dim),
            nn.Linear(self.teacher_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.visual_dim),
        )
        self.gate = nn.Parameter(torch.zeros(self.visual_dim))

    def _select_aux_tokens(
        self, aux_tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.max_aux_tokens <= 0 or aux_tokens.shape[1] <= self.max_aux_tokens:
            return aux_tokens, mask
        idx = (
            torch.linspace(0, aux_tokens.shape[1] - 1, steps=self.max_aux_tokens, device=aux_tokens.device)
            .round()
            .long()
        )
        selected_aux = aux_tokens.index_select(dim=1, index=idx)
        selected_mask = None if mask is None else mask.index_select(dim=1, index=idx)
        return selected_aux, selected_mask

    def forward(
        self,
        teacher_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        select_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if teacher_tokens.ndim != 3:
            raise ValueError(f"teacher_tokens must be rank 3, got shape {tuple(teacher_tokens.shape)}")
        if teacher_tokens.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"teacher_tokens last dim {teacher_tokens.shape[-1]} does not match teacher_dim={self.teacher_dim}"
            )
        if token_mask is not None and token_mask.shape != teacher_tokens.shape[:2]:
            raise ValueError(
                f"token_mask shape {tuple(token_mask.shape)} must match teacher token first dims "
                f"{tuple(teacher_tokens.shape[:2])}"
            )

        action_residual = self.to_action_residual(teacher_tokens.float())
        if action_residual.shape != (teacher_tokens.shape[0], teacher_tokens.shape[1], self.visual_dim):
            raise RuntimeError(
                f"action_residual shape {tuple(action_residual.shape)} does not match expected "
                f"{(teacher_tokens.shape[0], teacher_tokens.shape[1], self.visual_dim)}"
            )
        gate = (
            torch.tanh(self.gate)
            .to(dtype=action_residual.dtype, device=action_residual.device)
            .view(1, 1, self.visual_dim)
        )
        aux_tokens = action_residual * gate
        if token_mask is not None:
            aux_tokens = aux_tokens * token_mask.to(dtype=aux_tokens.dtype, device=aux_tokens.device).unsqueeze(-1)

        return self._select_aux_tokens(aux_tokens, token_mask) if select_aux else (aux_tokens, token_mask)


class Scheme3D3DeltaSuffixFusion(nn.Module):
    """Zero-gated residual update from predicted delta tokens into suffix states.

    Shape contract:
    - suffix_out: [B, H_action, D_suffix]
    - delta_tokens: [B, M_delta, D_delta]
    - output: [B, H_action, D_suffix]
    """

    def __init__(
        self,
        suffix_dim: int,
        delta_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.suffix_dim = int(suffix_dim)
        self.delta_dim = int(delta_dim)
        self.delta_to_suffix = nn.Linear(delta_dim, suffix_dim)
        self.cross_attn = _geo_modules.CrossAttentionBlock(suffix_dim, num_heads, dropout)
        self.delta_update = nn.Sequential(
            nn.LayerNorm(suffix_dim),
            nn.Linear(suffix_dim, suffix_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(suffix_dim * 2, suffix_dim),
        )
        self.gate = nn.Parameter(torch.zeros(suffix_dim))

    def forward(
        self,
        suffix_out: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        residual_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if suffix_out.ndim != 3:
            raise ValueError(f"suffix_out must be rank 3, got shape {tuple(suffix_out.shape)}")
        if delta_tokens.ndim != 3:
            raise ValueError(f"delta_tokens must be rank 3, got shape {tuple(delta_tokens.shape)}")
        if suffix_out.shape[0] != delta_tokens.shape[0]:
            raise ValueError(
                f"suffix_out batch {suffix_out.shape[0]} does not match delta_tokens batch {delta_tokens.shape[0]}"
            )
        if suffix_out.shape[-1] != self.suffix_dim:
            raise ValueError(f"suffix_out last dim {suffix_out.shape[-1]} does not match suffix_dim={self.suffix_dim}")
        if delta_tokens.shape[-1] != self.delta_dim:
            raise ValueError(
                f"delta_tokens last dim {delta_tokens.shape[-1]} does not match delta_dim={self.delta_dim}"
            )
        if isinstance(residual_scale, torch.Tensor):
            if residual_scale.shape != (suffix_out.shape[0],):
                raise ValueError(
                    "tensor residual_scale must have shape [B], got "
                    f"{tuple(residual_scale.shape)} for batch {suffix_out.shape[0]}"
                )
            if not torch.isfinite(residual_scale).all() or torch.any(residual_scale < 0.0):
                raise ValueError("tensor residual_scale must contain finite, non-negative values")
            residual_scale = residual_scale.to(device=suffix_out.device, dtype=torch.float32).view(-1, 1, 1)
        else:
            residual_scale = float(residual_scale)
            if not math.isfinite(residual_scale) or residual_scale < 0.0:
                raise ValueError(f"residual_scale must be finite and non-negative, got {residual_scale}")

        memory = self.delta_to_suffix(delta_tokens.float())
        fused = self.cross_attn(suffix_out.float(), memory)
        update = self.delta_update(fused)
        gate = torch.tanh(self.gate).to(dtype=update.dtype, device=update.device).view(1, 1, self.suffix_dim)
        # The scalar is explicit; no batch/timestep tensor broadcasting is used.
        return suffix_out.float() + residual_scale * gate * update


class Scheme3D3ActionDeltaAdapter(nn.Module):
    """Predict action-conditioned delta tokens and fuse them into suffix states.

    Shape contract:
    - suffix_out: [B, H_action, D_suffix]
    - current_tokens: [B, N_current, D_teacher]
    - estimated_actions: [B, H_action, A_dim]
    - state: [B, D_state]
    - token_mask: optional [B, N_current]
    - refined_suffix: [B, H_action, D_suffix]
    - pred_delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(
        self,
        *,
        current_dim: int,
        suffix_dim: int,
        delta_dim: int,
        num_delta_tokens: int,
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_horizon: int,
        horizon_id: int | None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if horizon_id is not None and (int(horizon_id) < 0 or int(horizon_id) >= int(max_horizon)):
            raise ValueError(f"horizon_id must be in [0, {max_horizon}), got {horizon_id}")
        self.horizon_id = None if horizon_id is None else int(horizon_id)
        self.predictor = DeltaActionPredictor(
            DeltaActionPredictorConfig(
                teacher_dim=current_dim,
                delta_dim=delta_dim,
                num_delta_tokens=num_delta_tokens,
                action_horizon=action_horizon,
                action_dim=action_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                max_horizon=max_horizon,
                zero_init_output=False,
            )
        )
        self.suffix_fusion = Scheme3D3DeltaSuffixFusion(
            suffix_dim=suffix_dim,
            delta_dim=delta_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def _horizon_ids(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.horizon_id is None:
            return None
        return torch.full((batch_size,), self.horizon_id, dtype=torch.long, device=device)

    def forward(
        self,
        suffix_out: torch.Tensor,
        current_tokens: torch.Tensor,
        estimated_actions: torch.Tensor,
        *,
        state: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if suffix_out.ndim != 3:
            raise ValueError(f"suffix_out must be rank 3, got shape {tuple(suffix_out.shape)}")
        if current_tokens.ndim != 3:
            raise ValueError(f"current_tokens must be rank 3, got shape {tuple(current_tokens.shape)}")
        if estimated_actions.ndim != 3:
            raise ValueError(f"estimated_actions must be rank 3, got shape {tuple(estimated_actions.shape)}")
        batch_size = int(suffix_out.shape[0])
        if current_tokens.shape[0] != batch_size:
            raise ValueError(
                f"current_tokens batch {current_tokens.shape[0]} does not match suffix_out batch {batch_size}"
            )
        if estimated_actions.shape[0] != batch_size:
            raise ValueError(
                f"estimated_actions batch {estimated_actions.shape[0]} does not match suffix_out batch {batch_size}"
            )
        if state.shape[0] != batch_size:
            raise ValueError(f"state batch {state.shape[0]} does not match suffix_out batch {batch_size}")

        horizon_ids = self._horizon_ids(batch_size, current_tokens.device)
        pred_delta_tokens = self.predictor(
            current_tokens,
            estimated_actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
        )
        refined_suffix = self.suffix_fusion(suffix_out, pred_delta_tokens)
        return refined_suffix, pred_delta_tokens

    def gate_abs_mean(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().mean()

    def gate_max_abs(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().max()


class Scheme3DeployableCurrentAdapter(nn.Module):
    """Project deployable pi visual tokens into frozen-tokenizer teacher space.

    Shape contract:
    - vision_hidden: [B, N_visual, D_visual]
    - token_mask: optional [B, N_visual]
    - current_tokens: [B, N_visual, D_teacher]
    """

    def __init__(self, visual_dim: int, teacher_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.teacher_dim = int(teacher_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.teacher_dim),
        )

    def forward(self, vision_hidden: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if vision_hidden.ndim != 3:
            raise ValueError(f"vision_hidden must be rank 3, got shape {tuple(vision_hidden.shape)}")
        if vision_hidden.shape[-1] != self.visual_dim:
            raise ValueError(f"vision_hidden last dim {vision_hidden.shape[-1]} does not match {self.visual_dim}")
        if token_mask is not None and token_mask.shape != vision_hidden.shape[:2]:
            raise ValueError(
                f"token_mask shape {tuple(token_mask.shape)} must match vision tokens {tuple(vision_hidden.shape[:2])}"
            )
        current_tokens = self.net(vision_hidden.float())
        expected = (vision_hidden.shape[0], vision_hidden.shape[1], self.teacher_dim)
        if current_tokens.shape != expected:
            raise RuntimeError(f"current_tokens shape {tuple(current_tokens.shape)} does not match {expected}")
        if token_mask is not None:
            current_tokens = current_tokens * token_mask.to(
                dtype=current_tokens.dtype, device=current_tokens.device
            ).unsqueeze(-1)
        return current_tokens


class Scheme3DeltaQueryPredictor(nn.Module):
    """Predict deployable delta tokens with learned queries cross-attending to pi/spatial tokens.

    Shape contract:
    - source_tokens: [B, N_source, D_source]
    - token_mask: optional [B, N_source]
    - language_features: [B, D_language]
    - state: [B, D_state]
    - action_intent: optional [B, D_action_intent]
    - pred_delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(
        self,
        *,
        source_dim: int,
        language_dim: int,
        state_dim: int,
        hidden_dim: int,
        delta_dim: int,
        num_delta_tokens: int,
        num_layers: int,
        num_heads: int,
        max_horizon: int,
        horizon_id: int | None,
        action_intent_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if horizon_id is not None and (int(horizon_id) < 0 or int(horizon_id) >= int(max_horizon)):
            raise ValueError(f"horizon_id must be in [0, {max_horizon}), got {horizon_id}")
        self.source_dim = int(source_dim)
        self.language_dim = int(language_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_dim = int(delta_dim)
        self.num_delta_tokens = int(num_delta_tokens)
        self.action_intent_dim = None if action_intent_dim is None else int(action_intent_dim)
        self.horizon_id = None if horizon_id is None else int(horizon_id)
        self.max_horizon = int(max_horizon)
        self.source_proj = nn.Sequential(
            nn.LayerNorm(self.source_dim),
            nn.Linear(self.source_dim, self.hidden_dim),
        )
        self.language_proj = nn.Sequential(
            nn.LayerNorm(self.language_dim),
            nn.Linear(self.language_dim, self.hidden_dim),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
        )
        self.action_intent_proj = (
            None
            if self.action_intent_dim is None
            else nn.Sequential(
                nn.LayerNorm(self.action_intent_dim),
                nn.Linear(self.action_intent_dim, self.hidden_dim),
            )
        )
        self.horizon_embed = nn.Embedding(self.max_horizon, self.hidden_dim)
        self.delta_queries = nn.Parameter(
            torch.randn(self.num_delta_tokens, self.hidden_dim) / math.sqrt(self.hidden_dim)
        )
        self.query_layers = nn.ModuleList(
            [_geo_modules.CrossAttentionBlock(self.hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.delta_dim),
        )

    def _horizon_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.horizon_id is None:
            return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)
        ids = torch.full((batch_size,), self.horizon_id, dtype=torch.long, device=device)
        return self.horizon_embed(ids).float()

    def forward(
        self,
        source_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must be rank 3, got shape {tuple(source_tokens.shape)}")
        batch_size, num_tokens, source_dim = source_tokens.shape
        if source_dim != self.source_dim:
            raise ValueError(f"source_tokens last dim {source_dim} does not match source_dim={self.source_dim}")
        if token_mask is not None and token_mask.shape != (batch_size, num_tokens):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} must match {(batch_size, num_tokens)}")
        if language_features.shape != (batch_size, self.language_dim):
            raise ValueError(
                f"language_features shape {tuple(language_features.shape)} must match {(batch_size, self.language_dim)}"
            )
        if state.shape != (batch_size, self.state_dim):
            raise ValueError(f"state shape {tuple(state.shape)} must match {(batch_size, self.state_dim)}")
        if self.action_intent_proj is None:
            if action_intent is not None:
                raise ValueError("action_intent was provided but this predictor has no action_intent_dim")
        else:
            if action_intent is None:
                raise ValueError("action_intent is required when action_intent_dim is configured")
            if action_intent.shape != (batch_size, self.action_intent_dim):
                raise ValueError(
                    f"action_intent shape {tuple(action_intent.shape)} must match "
                    f"{(batch_size, self.action_intent_dim)}"
                )

        memory = self.source_proj(source_tokens.float())
        condition = (
            self.language_proj(language_features.float())
            + self.state_proj(state.float())
            + self._horizon_features(batch_size, source_tokens.device)
        )
        if self.action_intent_proj is not None:
            condition = condition + self.action_intent_proj(action_intent.float())
        queries = self.delta_queries.to(dtype=memory.dtype, device=memory.device).unsqueeze(0)
        queries = queries.expand(batch_size, self.num_delta_tokens, self.hidden_dim) + condition.unsqueeze(1)
        for layer in self.query_layers:
            queries = layer(queries, memory, token_mask)
        pred_delta_tokens = self.delta_head(queries)
        expected = (batch_size, self.num_delta_tokens, self.delta_dim)
        if pred_delta_tokens.shape != expected:
            raise RuntimeError(f"pred_delta_tokens shape {tuple(pred_delta_tokens.shape)} does not match {expected}")
        return pred_delta_tokens


class Scheme3CorrectedDeltaAdapter(nn.Module):
    """Corrected Delta Align adapter: query predictor plus zero-gated suffix fusion."""

    def __init__(
        self,
        *,
        source_dim: int,
        language_dim: int,
        state_dim: int,
        suffix_dim: int,
        hidden_dim: int,
        delta_dim: int,
        num_delta_tokens: int,
        num_layers: int,
        num_heads: int,
        max_horizon: int,
        horizon_id: int | None,
        action_intent_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.predictor = Scheme3DeltaQueryPredictor(
            source_dim=source_dim,
            language_dim=language_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            delta_dim=delta_dim,
            num_delta_tokens=num_delta_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            max_horizon=max_horizon,
            horizon_id=horizon_id,
            action_intent_dim=action_intent_dim,
            dropout=dropout,
        )
        self.suffix_fusion = Scheme3D3DeltaSuffixFusion(
            suffix_dim=suffix_dim,
            delta_dim=delta_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        suffix_out: torch.Tensor,
        current_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_delta_tokens = self.predictor(
            current_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        refined_suffix = self.suffix_fusion(suffix_out, pred_delta_tokens)
        return refined_suffix, pred_delta_tokens

    def gate_abs_mean(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().mean()

    def gate_max_abs(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().max()


class Scheme4LanguageActionDeltaAdapter(nn.Module):
    """Scheme4 LAC delta predictor plus zero-gated suffix fusion.

    Shape contract:
    - suffix_out: [B, H_action, D_suffix]
    - spatial_tokens S_t: [B, N_visual, D_spatial]
    - token_mask: optional [B, N_visual]
    - language_features: [B, D_language]
    - state: [B, D_state]
    - action_intent: [B, D_action_intent]
    - pred_delta_tokens: [B, M_delta, D_delta]
    - delta_residual: [B, H_action, D_suffix]

    ``delta_enabled=False`` is the scheme4 delta-disabled sanity path. It
    returns identity suffix features and same-shaped zero delta tokens, so the
    surrounding policy should match the scheme2_unified path as closely as
    possible under the scheme4 code path.
    """

    SUPPORTED_ACTION_INTENT_MODES = (
        "t1_suffix_mean",
        "mid_denoise_hidden",
        "two_pass_preliminary",
        "endpoint_estimated_action",
    )
    SUPPORTED_FUSION_TYPES = ("suffix_cross_attn", "mid_layer_cross_attn", "late_layer_cross_attn", "velocity_residual")

    def __init__(
        self,
        *,
        spatial_dim: int,
        language_dim: int,
        state_dim: int,
        action_intent_dim: int,
        suffix_dim: int,
        hidden_dim: int,
        delta_dim: int,
        num_delta_tokens: int,
        num_layers: int,
        num_heads: int,
        max_horizon: int,
        horizon_id: int | None,
        action_intent_mode: str = "t1_suffix_mean",
        delta_fusion_type: str = "suffix_cross_attn",
        delta_enabled: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.language_dim = int(language_dim)
        self.state_dim = int(state_dim)
        self.action_intent_dim = int(action_intent_dim)
        self.suffix_dim = int(suffix_dim)
        self.delta_dim = int(delta_dim)
        self.num_delta_tokens = int(num_delta_tokens)
        self.action_intent_mode = str(action_intent_mode)
        self.delta_fusion_type = str(delta_fusion_type)
        self.delta_enabled = bool(delta_enabled)

        if self.action_intent_mode not in self.SUPPORTED_ACTION_INTENT_MODES:
            raise ValueError(
                f"scheme4 action_intent_mode must be one of {self.SUPPORTED_ACTION_INTENT_MODES}, "
                f"got {self.action_intent_mode!r}"
            )
        if self.action_intent_mode != "t1_suffix_mean":
            raise NotImplementedError(
                f"scheme4 action_intent_mode={self.action_intent_mode!r} is reserved but not implemented yet"
            )
        if self.delta_fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"scheme4 delta_fusion_type must be one of {self.SUPPORTED_FUSION_TYPES}, "
                f"got {self.delta_fusion_type!r}"
            )
        if self.delta_fusion_type != "suffix_cross_attn":
            raise NotImplementedError(
                f"scheme4 delta_fusion_type={self.delta_fusion_type!r} is reserved but not implemented yet"
            )

        self.predictor = Scheme3DeltaQueryPredictor(
            source_dim=self.spatial_dim,
            language_dim=self.language_dim,
            state_dim=self.state_dim,
            hidden_dim=hidden_dim,
            delta_dim=self.delta_dim,
            num_delta_tokens=self.num_delta_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            max_horizon=max_horizon,
            horizon_id=horizon_id,
            action_intent_dim=self.action_intent_dim,
            dropout=dropout,
        )
        self.suffix_fusion = Scheme3D3DeltaSuffixFusion(
            suffix_dim=self.suffix_dim,
            delta_dim=self.delta_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        if not self.delta_enabled:
            _set_module_trainable(self, False)

    def _validate_condition_shapes(
        self,
        spatial_tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> int:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be rank 3, got shape {tuple(spatial_tokens.shape)}")
        batch_size, num_tokens, spatial_dim = spatial_tokens.shape
        if spatial_dim != self.spatial_dim:
            raise ValueError(f"spatial_tokens last dim {spatial_dim} does not match spatial_dim={self.spatial_dim}")
        if token_mask is not None and token_mask.shape != (batch_size, num_tokens):
            raise ValueError(f"token_mask shape {tuple(token_mask.shape)} must match {(batch_size, num_tokens)}")
        if language_features.shape != (batch_size, self.language_dim):
            raise ValueError(
                f"language_features shape {tuple(language_features.shape)} must match {(batch_size, self.language_dim)}"
            )
        if state.shape != (batch_size, self.state_dim):
            raise ValueError(f"state shape {tuple(state.shape)} must match {(batch_size, self.state_dim)}")
        if action_intent.shape != (batch_size, self.action_intent_dim):
            raise ValueError(
                f"action_intent shape {tuple(action_intent.shape)} must match {(batch_size, self.action_intent_dim)}"
            )
        return batch_size

    def _zero_delta_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.num_delta_tokens,
            self.delta_dim,
            dtype=dtype,
            device=device,
        )

    def predict_delta(
        self,
        spatial_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self._validate_condition_shapes(
            spatial_tokens,
            token_mask,
            language_features,
            state,
            action_intent,
        )
        if not self.delta_enabled:
            return self._zero_delta_tokens(batch_size, spatial_tokens.device, spatial_tokens.float().dtype)
        return self.predictor(
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    def forward(
        self,
        suffix_out: torch.Tensor,
        spatial_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
        residual_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if suffix_out.ndim != 3:
            raise ValueError(f"suffix_out must be rank 3, got shape {tuple(suffix_out.shape)}")
        if suffix_out.shape[-1] != self.suffix_dim:
            raise ValueError(f"suffix_out last dim {suffix_out.shape[-1]} does not match suffix_dim={self.suffix_dim}")
        pred_delta_tokens = self.predict_delta(
            spatial_tokens,
            token_mask=token_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        if not self.delta_enabled:
            refined_suffix = suffix_out.float()
        else:
            refined_suffix = self.suffix_fusion(
                suffix_out,
                pred_delta_tokens,
                residual_scale=residual_scale,
            )
        delta_residual = refined_suffix.float() - suffix_out.float()
        expected = suffix_out.shape
        if refined_suffix.shape != expected:
            raise RuntimeError(f"refined_suffix shape {tuple(refined_suffix.shape)} does not match {tuple(expected)}")
        if delta_residual.shape != expected:
            raise RuntimeError(f"delta_residual shape {tuple(delta_residual.shape)} does not match {tuple(expected)}")
        return refined_suffix, pred_delta_tokens, delta_residual

    def gate_abs_mean(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().mean()

    def gate_max_abs(self) -> torch.Tensor:
        return self.suffix_fusion.gate.float().abs().max()


SCHEME4_INFERENCE_CONDITION_ABLATIONS = (
    "none",
    "zero_language",
    "zero_action",
    "zero_language_action",
    "wrong_language",
    "wrong_action",
    "shuffled_language",
    "shuffled_action",
)

SCHEME4_TRAINING_ACTION_CONDITIONS = (
    "normal",
    "zero_action",
)

SCHEME4_INFERENCE_DELTA_CONTROLS = (
    "normal",
    "zero_delta",
    "mean_delta",
    "random_delta",
    "norm_matched_random_delta",
    "cross_episode_delta",
    "batch_shuffled_delta",
)


def _fallback_previous_or_zeros(current: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
    """Use a cached previous tensor when its full shape matches, otherwise zeros."""
    if previous is None or previous.shape != current.shape:
        return torch.zeros_like(current)
    return previous.to(device=current.device, dtype=current.dtype)


def apply_scheme4_training_action_condition(mode: str, action_intent: torch.Tensor) -> torch.Tensor:
    """Return the action condition used by a consistently trained Scheme4 arm.

    Shape contract:
    - action_intent: [B, D_suffix]

    This is intentionally distinct from inference-only perturbation controls.
    ``zero_action`` is applied in both ``forward`` and ``sample_actions`` so a
    matched control never sees an action-derived predictor input at training or
    deployment. The original pi0.5 action-expert path remains unchanged.
    """

    if mode not in SCHEME4_TRAINING_ACTION_CONDITIONS:
        raise ValueError(
            f"scheme4_training_action_condition must be one of {SCHEME4_TRAINING_ACTION_CONDITIONS}, got {mode!r}"
        )
    if action_intent.ndim != 2:
        raise ValueError(f"action_intent must be rank 2, got shape {tuple(action_intent.shape)}")
    if mode == "normal":
        return action_intent
    return torch.zeros_like(action_intent)


def apply_scheme4_inference_condition_ablation(
    mode: str,
    language_features: torch.Tensor,
    action_intent: torch.Tensor,
    *,
    previous_language_features: torch.Tensor | None = None,
    previous_action_intent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return delta-predictor conditions after an inference-only ablation.

    Shapes:
    - language_features: [B, D_language]
    - action_intent: [B, D_action_intent]
    - previous_language_features: optional [B, D_language]
    - previous_action_intent: optional [B, D_action_intent]
    """
    if mode not in SCHEME4_INFERENCE_CONDITION_ABLATIONS:
        raise ValueError(
            f"scheme4_inference_condition_ablation must be one of {SCHEME4_INFERENCE_CONDITION_ABLATIONS}, got {mode!r}"
        )
    if language_features.ndim != 2:
        raise ValueError(f"language_features must be rank 2, got shape {tuple(language_features.shape)}")
    if action_intent.ndim != 2:
        raise ValueError(f"action_intent must be rank 2, got shape {tuple(action_intent.shape)}")
    if language_features.shape[0] != action_intent.shape[0]:
        raise ValueError(
            "language_features and action_intent must have the same batch size, got "
            f"{language_features.shape[0]} and {action_intent.shape[0]}"
        )

    ablated_language = language_features
    ablated_action = action_intent
    if mode in {"zero_language", "zero_language_action"}:
        ablated_language = torch.zeros_like(language_features)
    if mode in {"zero_action", "zero_language_action"}:
        ablated_action = torch.zeros_like(action_intent)
    if mode == "wrong_language":
        ablated_language = _fallback_previous_or_zeros(language_features, previous_language_features)
    if mode == "wrong_action":
        ablated_action = _fallback_previous_or_zeros(action_intent, previous_action_intent)
    if mode == "shuffled_language":
        if language_features.shape[0] > 1:
            ablated_language = torch.roll(language_features, shifts=1, dims=0)
        else:
            ablated_language = _fallback_previous_or_zeros(language_features, previous_language_features)
    if mode == "shuffled_action":
        if action_intent.shape[0] > 1:
            ablated_action = torch.roll(action_intent, shifts=1, dims=0)
        else:
            ablated_action = _fallback_previous_or_zeros(action_intent, previous_action_intent)
    return ablated_language, ablated_action


def _prepare_mean_delta_tokens(
    mean_delta_tokens: torch.Tensor,
    expected_shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, num_delta_tokens, delta_dim = expected_shape
    if mean_delta_tokens.ndim == 2:
        if mean_delta_tokens.shape != (num_delta_tokens, delta_dim):
            raise ValueError(
                f"mean_delta_tokens shape {tuple(mean_delta_tokens.shape)} must match {(num_delta_tokens, delta_dim)}"
            )
        mean_delta_tokens = mean_delta_tokens.unsqueeze(0)
    elif mean_delta_tokens.ndim == 3:
        if mean_delta_tokens.shape[1:] != (num_delta_tokens, delta_dim):
            raise ValueError(
                f"mean_delta_tokens shape {tuple(mean_delta_tokens.shape)} must have trailing shape "
                f"{(num_delta_tokens, delta_dim)}"
            )
    else:
        raise ValueError(f"mean_delta_tokens must be rank 2 or 3, got shape {tuple(mean_delta_tokens.shape)}")

    if mean_delta_tokens.shape[0] == batch_size:
        return mean_delta_tokens.to(device=device, dtype=dtype)
    if mean_delta_tokens.shape[0] == 1:
        return mean_delta_tokens.to(device=device, dtype=dtype).expand(batch_size, num_delta_tokens, delta_dim).clone()
    raise ValueError(
        f"mean_delta_tokens batch {mean_delta_tokens.shape[0]} must be 1 or match pred_delta batch {batch_size}"
    )


def apply_scheme4_inference_delta_control(
    mode: str,
    pred_delta_tokens: torch.Tensor,
    *,
    mean_delta_tokens: torch.Tensor | None = None,
    previous_delta_tokens: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return inference-control delta tokens with the same shape as prediction.

    Shape contract:
    - pred_delta_tokens: [B, M_delta, D_delta]
    - mean_delta_tokens: optional [M_delta, D_delta], [1, M_delta, D_delta], or [B, M_delta, D_delta]
    - previous_delta_tokens: optional [B, M_delta, D_delta]
    """
    if mode not in SCHEME4_INFERENCE_DELTA_CONTROLS:
        raise ValueError(
            f"scheme4_inference_delta_control must be one of {SCHEME4_INFERENCE_DELTA_CONTROLS}, got {mode!r}"
        )
    if pred_delta_tokens.ndim != 3:
        raise ValueError(f"pred_delta_tokens must be rank 3, got shape {tuple(pred_delta_tokens.shape)}")

    expected_shape = tuple(int(dim) for dim in pred_delta_tokens.shape)
    if mode == "normal":
        controlled = pred_delta_tokens
    elif mode == "zero_delta":
        controlled = torch.zeros_like(pred_delta_tokens)
    elif mode == "mean_delta":
        if mean_delta_tokens is None:
            raise ValueError("mean_delta control requires scheme4_inference_delta_mean_path or mean_delta_tokens")
        controlled = _prepare_mean_delta_tokens(
            mean_delta_tokens,
            expected_shape,
            device=pred_delta_tokens.device,
            dtype=pred_delta_tokens.dtype,
        )
    elif mode == "random_delta":
        controlled = torch.randn(
            expected_shape,
            device=pred_delta_tokens.device,
            dtype=torch.float32,
            generator=generator,
        ).to(dtype=pred_delta_tokens.dtype)
    elif mode == "norm_matched_random_delta":
        random_tokens = torch.randn(
            expected_shape,
            device=pred_delta_tokens.device,
            dtype=torch.float32,
            generator=generator,
        )
        random_norm = random_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_norm = pred_delta_tokens.float().norm(dim=-1, keepdim=True)
        controlled = (random_tokens / random_norm * target_norm).to(dtype=pred_delta_tokens.dtype)
    elif mode == "cross_episode_delta":
        controlled = _fallback_previous_or_zeros(pred_delta_tokens, previous_delta_tokens)
    elif mode == "batch_shuffled_delta":
        if pred_delta_tokens.shape[0] > 1:
            controlled = torch.roll(pred_delta_tokens, shifts=1, dims=0)
        else:
            controlled = _fallback_previous_or_zeros(pred_delta_tokens, previous_delta_tokens)
    else:
        raise AssertionError(f"unhandled scheme4_inference_delta_control={mode!r}")

    if controlled.shape != pred_delta_tokens.shape:
        raise RuntimeError(
            f"controlled delta shape {tuple(controlled.shape)} must match pred_delta_tokens "
            f"{tuple(pred_delta_tokens.shape)}"
        )
    if not torch.isfinite(controlled.float()).all():
        raise RuntimeError(f"controlled delta contains non-finite values for mode {mode!r}")
    return controlled


class Scheme1SharedMLPPI0Pytorch(GeometryResidualPI0Pytorch):
    """Scheme 1: shared post-backbone MLP with action and spatial heads."""

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        _use_eager_sample_actions(self)
        visual_dim = self.LLM_width
        suffix_dim = self.action_out_proj.in_features
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)

        self.scheme1_adapter = Scheme1SharedMLPSpatialActionAdapter(
            visual_dim=visual_dim,
            suffix_dim=suffix_dim,
            teacher_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            dropout=proposal_dropout,
        )

        # Scheme 1 uses a direct spatial head instead of the slot decoder.
        _set_module_trainable(self.teacher_token_decoder, False)

    def _scheme1_compute_refined_visual(
        self,
        vision_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheme1_adapter.refine_visual(vision_hidden)

    def _scheme1_apply_action_head(self, suffix_out: torch.Tensor) -> torch.Tensor:
        refined_suffix = self.scheme1_adapter.refine_suffix(suffix_out)
        return self._apply_checkpoint(lambda x: self.action_out_proj(x), refined_suffix)

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            future_teacher_images,
            future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _, all_hidden_states = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=True,
            )
            return suffix_out, all_hidden_states

        suffix_out, all_hidden_states = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        base_v_t = self._scheme1_apply_action_head(suffix_out)

        (prefix_hidden, _) = all_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)

        refined_visual = self._scheme1_compute_refined_visual(vision_hidden.float())
        pred_current_hidden = self.scheme1_adapter.predict_spatial(refined_visual)
        current_geo_loss = _geo_modules.masked_cosine_loss(pred_current_hidden, teacher_current_hidden, vision_mask)

        z_geo, z_act = self._compute_geo_conditioning(
            prefix_hidden,
            refined_visual,
            vision_mask,
            x_t,
            time,
            prefix_mask=prefix_pad_masks,
        )

        teacher_delta = (teacher_future_hidden - teacher_current_hidden).detach()
        pred_delta = self.geometry_variation_head(z_geo, z_act, teacher_delta.shape[1])
        if future_teacher_fallback is None:
            delta_valid_mask = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
        else:
            delta_valid_mask = ~future_teacher_fallback.to(device=actions.device).bool().view(actions.shape[0])
        delta_geo_loss, delta_metrics = _geo_modules.geometry_delta_loss(
            pred_delta,
            teacher_delta,
            token_mask=vision_mask,
            valid_batch_mask=delta_valid_mask,
            norm_coeff=self.delta_norm_coeff,
            global_coeff=self.delta_global_coeff,
            motion_temp=self.delta_motion_temp,
            min_teacher_norm=self.delta_min_teacher_norm,
        )

        v_t, residual = self._apply_residual_flow(suffix_out, base_v_t.float(), x_t, z_geo, z_act)
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()
        residual_reg = residual.float().pow(2).mean()
        loss = (
            flow_loss
            + self.current_geo_loss_coeff * current_geo_loss
            + self.delta_loss_coeff * delta_geo_loss
            + self.residual_reg_coeff * residual_reg
        )

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "delta_geo_loss": delta_geo_loss,
            "residual_reg": residual_reg,
            "residual_abs_mean": residual.float().abs().mean().detach(),
            "scheme1_refined_visual_norm": refined_visual.float().norm(dim=-1).mean().detach(),
        }
        output.update(delta_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values, prefix_hidden_states = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        refined_visual = self._scheme1_compute_refined_visual(vision_hidden.float())
        z_geo = self.geo_adapter(refined_visual.float(), vision_mask)
        geo_cache = {"prefix_hidden": prefix_hidden.float(), "z_geo": z_geo}

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        base_v_t = self._scheme1_apply_action_head(suffix_out)
        z_act = self.action_intent_adapter(
            x_t.float(),
            self._time_embedding_for_geo(timestep),
            geo_cache["prefix_hidden"].float(),
            geo_cache["z_geo"],
            prefix_mask=prefix_pad_masks,
        )
        v_t, _ = self._apply_residual_flow(suffix_out, base_v_t.float(), x_t, geo_cache["z_geo"], z_act)
        return v_t


class Scheme1ResidualQueryPI0Pytorch(GeometryResidualPI0Pytorch):
    """Scheme1-RQ: residual query adapter for a safer Scheme 1 retry."""

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        _use_eager_sample_actions(self)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)

        self.scheme1_rq_adapter = Scheme1ResidualQueryAdapter(
            visual_dim=self.LLM_width,
            suffix_dim=self.action_out_proj.in_features,
            teacher_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            num_queries=self.geo_num_slots,
            num_heads=self.geo_num_heads,
            max_tokens=self.geo_max_tokens,
            dropout=proposal_dropout,
        )

        # This variant tests a cleaner Scheme 1 path and does not use the older
        # residual-flow geometry modules.
        _set_module_trainable(self.geo_adapter, False)
        _set_module_trainable(self.teacher_token_decoder, False)
        _set_module_trainable(self.action_intent_adapter, False)
        _set_module_trainable(self.geometry_variation_head, False)
        _set_module_trainable(self.residual_flow_expert, False)

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _, all_hidden_states = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=True,
            )
            return suffix_out, all_hidden_states

        suffix_out, all_hidden_states = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)

        (prefix_hidden, _) = all_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        if teacher_current_hidden.shape[-1] != self.teacher_dim:
            raise RuntimeError(
                f"VGGT teacher dim {teacher_current_hidden.shape[-1]} does not match geo_teacher_dim={self.teacher_dim}."
            )

        _refined_visual, query_tokens = self.scheme1_rq_adapter.encode_visual(vision_hidden.float(), vision_mask)
        pred_current_hidden = self.scheme1_rq_adapter.predict_spatial(query_tokens, teacher_current_hidden.shape[1])
        current_geo_loss, spatial_metrics = _spatial_alignment_loss(
            pred_current_hidden, teacher_current_hidden, vision_mask
        )

        refined_suffix = self.scheme1_rq_adapter.refine_suffix(suffix_out, query_tokens)
        v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), refined_suffix)
        with torch.no_grad():
            base_v_t = self.action_out_proj(suffix_out)
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "scheme1_rq_query_norm": query_tokens.float().norm(dim=-1).mean().detach(),
            "scheme1_rq_gate_abs_mean": self.scheme1_rq_adapter.suffix_fusion.gate.float().abs().mean().detach(),
            "current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
        }
        output.update(spatial_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values, prefix_hidden_states = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        _, query_tokens = self.scheme1_rq_adapter.encode_visual(vision_hidden.float(), vision_mask)
        geo_cache = {"query_tokens": query_tokens.float()}

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        refined_suffix = self.scheme1_rq_adapter.refine_suffix(suffix_out, geo_cache["query_tokens"])
        return self.action_out_proj(refined_suffix)


class Scheme2VisualConcatPI0Pytorch(GeometryResidualPI0Pytorch):
    """Scheme 2: VGGT-aligned visual auxiliary tokens concatenated to prefix."""

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        _use_eager_sample_actions(self)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        max_aux_tokens = int(getattr(extra_config, "scheme2_max_aux_tokens", 128))
        self.scheme2_injection_mode = str(getattr(extra_config, "scheme2_injection_mode", "residual_prefix"))
        if self.scheme2_injection_mode not in ("residual_prefix", "concat"):
            raise ValueError(
                f"scheme2_injection_mode must be 'residual_prefix' or 'concat', got {self.scheme2_injection_mode!r}"
            )
        self.scheme2_adapter = Scheme2VisualConcatAdapter(
            visual_dim=self.LLM_width,
            teacher_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            max_aux_tokens=max_aux_tokens,
            dropout=proposal_dropout,
        )

        # Scheme 2 routes geometry into the action expert through appended prefix
        # tokens, not through the residual-flow path.
        _set_module_trainable(self.geo_adapter, False)
        _set_module_trainable(self.teacher_token_decoder, False)
        _set_module_trainable(self.action_intent_adapter, False)
        _set_module_trainable(self.geometry_variation_head, False)
        _set_module_trainable(self.residual_flow_expert, False)

    def _prefix_hidden_from_embeddings(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        *,
        use_cache: bool,
        output_hidden_states: bool,
    ):
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        return self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

    def _append_aux_prefix_tokens(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        aux_tokens: torch.Tensor,
        aux_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_tokens = aux_tokens.to(dtype=prefix_embs.dtype, device=prefix_embs.device)
        if aux_mask is None:
            aux_pad = torch.ones(aux_tokens.shape[:2], dtype=torch.bool, device=prefix_embs.device)
        else:
            aux_pad = aux_mask.bool().to(device=prefix_embs.device)
        aux_att = torch.zeros(aux_tokens.shape[:2], dtype=prefix_att_masks.dtype, device=prefix_embs.device)
        return (
            torch.cat([prefix_embs, aux_tokens], dim=1),
            torch.cat([prefix_pad_masks, aux_pad], dim=1),
            torch.cat([prefix_att_masks, aux_att], dim=1),
        )

    def _inject_aux_prefix_tokens(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        aux_tokens: torch.Tensor,
        aux_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scheme2_injection_mode == "concat":
            return self._append_aux_prefix_tokens(prefix_embs, prefix_pad_masks, prefix_att_masks, aux_tokens, aux_mask)

        aux_tokens = aux_tokens.to(dtype=prefix_embs.dtype, device=prefix_embs.device)
        if aux_tokens.shape[1] > prefix_embs.shape[1]:
            raise ValueError(f"aux token length {aux_tokens.shape[1]} exceeds prefix length {prefix_embs.shape[1]}")
        if aux_mask is not None:
            aux_tokens = aux_tokens * aux_mask.to(dtype=aux_tokens.dtype, device=aux_tokens.device).unsqueeze(-1)
        prefix_embs_aug = prefix_embs.clone()
        prefix_embs_aug[:, : aux_tokens.shape[1], :] = prefix_embs_aug[:, : aux_tokens.shape[1], :] + aux_tokens
        return prefix_embs_aug, prefix_pad_masks, prefix_att_masks

    def _scheme2_spatial_loss(
        self,
        pred_current_hidden: torch.Tensor,
        teacher_current_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = _geo_modules.masked_cosine_loss(pred_current_hidden, teacher_current_hidden, vision_mask)
        return loss, {}

    def _scheme2_extra_metrics(self) -> dict[str, torch.Tensor]:
        return {}

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align]
            vision_hidden = prefix_hidden[:, :img_len, :].detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        current_geo_loss, spatial_metrics = self._scheme2_spatial_loss(
            pred_current_hidden, teacher_current_hidden, vision_mask
        )

        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs_aug = prefix_embs_aug.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks_aug, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_aug, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs_aug, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_aug, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs_aug,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)

        def action_out_func(suffix_tokens):
            return self.action_out_proj(suffix_tokens)

        v_t = self._apply_checkpoint(action_out_func, suffix_out)
        flow_loss = F.mse_loss(u_t, v_t)
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": flow_loss.detach(),
            "current_geo_loss": current_geo_loss,
            "scheme2_aux_abs_mean": aux_tokens.float().abs().mean().detach(),
            "scheme2_aux_tokens": torch.tensor(aux_tokens.shape[1], dtype=torch.float32, device=loss.device),
            "scheme2_injection_is_concat": torch.tensor(
                float(self.scheme2_injection_mode == "concat"), dtype=torch.float32, device=loss.device
            ),
        }
        output.update(spatial_metrics)
        output.update(self._scheme2_extra_metrics())
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        _, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs_aug,
            prefix_pad_masks_aug,
            prefix_att_masks_aug,
            use_cache=True,
            output_hidden_states=False,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks_aug, past_key_values, x_t, time.expand(bsize), None)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache=None):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class Scheme2GatedDualHeadPI0Pytorch(Scheme2VisualConcatPI0Pytorch):
    """Scheme2-G: gated dual-head refinement of the successful Scheme 2 path."""

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        max_aux_tokens = int(getattr(extra_config, "scheme2_max_aux_tokens", 128))
        self.scheme2_adapter = Scheme2GatedDualHeadAdapter(
            visual_dim=self.LLM_width,
            teacher_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            max_aux_tokens=max_aux_tokens,
            dropout=proposal_dropout,
        )

    def _scheme2_spatial_loss(
        self,
        pred_current_hidden: torch.Tensor,
        teacher_current_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return _spatial_alignment_loss(pred_current_hidden, teacher_current_hidden, vision_mask)

    def _scheme2_extra_metrics(self) -> dict[str, torch.Tensor]:
        gate = self.scheme2_adapter.gate.float()
        return {
            "scheme2_gate_abs_mean": gate.abs().mean().detach(),
            "scheme2_gate_max_abs": gate.abs().max().detach(),
        }


class Scheme2UnifiedSpatialPI0Pytorch(Scheme2VisualConcatPI0Pytorch):
    """Corrected Scheme2: unified spatial bottleneck plus zero-gated prefix residual.

    Unlike old Scheme2, the tensor supervised by current VGGT alignment is the
    same tensor that is projected into the action prefix path.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        max_aux_tokens = int(getattr(extra_config, "scheme2_max_aux_tokens", 128))
        self.scheme2_adapter = Scheme2UnifiedSpatialBottleneckAdapter(
            visual_dim=self.LLM_width,
            spatial_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            max_aux_tokens=max_aux_tokens,
            dropout=proposal_dropout,
        )

    def _scheme2_current_geo_loss(
        self,
        spatial_tokens: torch.Tensor,
        teacher_images: list[torch.Tensor] | None,
        vggt: nn.Module,
        vision_hidden_ref: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if float(self.current_geo_loss_coeff) == 0.0:
            zero = spatial_tokens.float().sum() * 0.0
            return zero, {}, torch.tensor(1.0, dtype=torch.float32, device=spatial_tokens.device)

        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden_ref)
        if teacher_current_hidden.shape != spatial_tokens.shape:
            raise RuntimeError(
                f"Unified Scheme2 requires teacher_current_hidden shape {tuple(teacher_current_hidden.shape)} "
                f"to match spatial_tokens {tuple(spatial_tokens.shape)}."
            )
        current_geo_loss, spatial_metrics = _spatial_alignment_loss(
            spatial_tokens,
            teacher_current_hidden.detach(),
            vision_mask,
        )
        return current_geo_loss, spatial_metrics, torch.tensor(0.0, dtype=torch.float32, device=spatial_tokens.device)

    def _scheme2_extra_metrics(self) -> dict[str, torch.Tensor]:
        gate = self.scheme2_adapter.gate.float()
        return {
            "scheme2_unified_gate_abs_mean": gate.abs().mean().detach(),
            "scheme2_unified_gate_max_abs": gate.abs().max().detach(),
        }

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align]
            vision_hidden = prefix_hidden[:, :img_len, :].detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        spatial_tokens, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        current_geo_loss, spatial_metrics, spatial_loss_skipped = self._scheme2_current_geo_loss(
            spatial_tokens,
            teacher_images,
            vggt,
            vision_hidden,
            vision_mask,
        )

        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs_aug = prefix_embs_aug.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks_aug, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_aug, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs_aug, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_aug, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs_aug,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        flow_loss = F.mse_loss(u_t, v_t)
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": flow_loss.detach(),
            "current_geo_loss": current_geo_loss,
            "scheme2_aux_abs_mean": aux_tokens.float().abs().mean().detach(),
            "scheme2_aux_tokens": torch.tensor(aux_tokens.shape[1], dtype=torch.float32, device=loss.device),
            "scheme2_injection_is_concat": torch.tensor(
                float(self.scheme2_injection_mode == "concat"), dtype=torch.float32, device=loss.device
            ),
            "scheme2_unified_spatial_norm": spatial_tokens.float().norm(dim=-1).mean().detach(),
            "scheme2_unified_spatial_loss_skipped": spatial_loss_skipped.detach(),
            "current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
        }
        output.update(spatial_metrics)
        output.update(self._scheme2_extra_metrics())
        return output


class Scheme2UnifiedSpatialAuxOnlyPI0Pytorch(Scheme2UnifiedSpatialPI0Pytorch):
    """Corrected Scheme2 auxiliary-only control.

    It trains the same unified spatial bottleneck with current VGGT loss but
    returns the original prefix embeddings unchanged, so action-path gains from
    explicit spatial injection are separated from auxiliary regularization.
    """

    def _inject_aux_prefix_tokens(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        aux_tokens: torch.Tensor,
        aux_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if aux_tokens.ndim != 3:
            raise ValueError(f"aux_tokens must be rank 3, got shape {tuple(aux_tokens.shape)}")
        if aux_tokens.shape[0] != prefix_embs.shape[0]:
            raise ValueError(
                f"aux_tokens batch {aux_tokens.shape[0]} does not match prefix batch {prefix_embs.shape[0]}"
            )
        if aux_tokens.shape[-1] != prefix_embs.shape[-1]:
            raise ValueError(f"aux_tokens dim {aux_tokens.shape[-1]} does not match prefix dim {prefix_embs.shape[-1]}")
        if aux_mask is not None and aux_mask.shape != aux_tokens.shape[:2]:
            raise ValueError(
                f"aux_mask shape {tuple(aux_mask.shape)} does not match aux_tokens {tuple(aux_tokens.shape[:2])}"
            )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _scheme2_extra_metrics(self) -> dict[str, torch.Tensor]:
        metrics = super()._scheme2_extra_metrics()
        gate = self.scheme2_adapter.gate.float()
        metrics["scheme2_unified_aux_only"] = torch.tensor(1.0, dtype=torch.float32, device=gate.device)
        return metrics


class DirectVGGTInjectionPI0Pytorch(Scheme2VisualConcatPI0Pytorch):
    """Online-teacher baseline: current VGGT tokens are injected into the prefix.

    This variant is not teacher-free.  It exists to measure the success/latency
    tradeoff of running VGGT at inference and directly projecting current VGGT
    tokens into the same zero-gated prefix-residual action path used by the
    Spatial Align variants.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        max_aux_tokens = int(getattr(extra_config, "scheme2_max_aux_tokens", 128))
        self.direct_vggt_adapter = DirectVGGTInjectionAdapter(
            teacher_dim=self.teacher_dim,
            visual_dim=self.LLM_width,
            hidden_dim=proposal_hidden_dim,
            max_aux_tokens=max_aux_tokens,
            dropout=proposal_dropout,
        )
        self.direct_vggt_weight_path = getattr(extra_config, "vggt_weight_path", None)
        self.direct_vggt_runtime = None

    def _get_direct_vggt_runtime(self, device: torch.device) -> nn.Module:
        runtime = self.direct_vggt_runtime
        if runtime is not None:
            return runtime.to(device).eval()

        if self.direct_vggt_weight_path is None:
            raise ValueError("direct_vggt_injection inference requires config.vggt_weight_path")

        from vggt.models.vggt import VGGT

        runtime = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        ).to(device)
        vggt_path = os.path.join(str(self.direct_vggt_weight_path), "model.pt")
        if not os.path.exists(vggt_path):
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_path}")
        runtime.load_state_dict(torch.load(vggt_path, map_location=device, weights_only=False), strict=False)
        runtime.eval()
        _set_module_trainable(runtime, False)
        self.direct_vggt_runtime = runtime
        return runtime

    def _direct_vggt_tokens(
        self,
        image_inputs: list[torch.Tensor],
        vggt: nn.Module,
        vision_hidden_ref: torch.Tensor,
    ) -> torch.Tensor:
        teacher_current_hidden = self._encode_vggt_hidden(image_inputs, vggt, vision_hidden_ref)
        if teacher_current_hidden.ndim != 3:
            raise RuntimeError(f"direct VGGT tokens must be rank 3, got shape {tuple(teacher_current_hidden.shape)}")
        if teacher_current_hidden.shape[-1] != self.teacher_dim:
            raise RuntimeError(
                f"direct VGGT dim {teacher_current_hidden.shape[-1]} does not match teacher_dim={self.teacher_dim}"
            )
        if teacher_current_hidden.shape[:2] != vision_hidden_ref.shape[:2]:
            raise RuntimeError(
                f"direct VGGT token shape {tuple(teacher_current_hidden.shape[:2])} must match "
                f"vision_hidden_ref {tuple(vision_hidden_ref.shape[:2])}"
            )
        return teacher_current_hidden.detach()

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align]
            vision_hidden = prefix_hidden[:, :img_len, :].detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        direct_images = teacher_images if teacher_images is not None else images
        teacher_current_hidden = self._direct_vggt_tokens(direct_images, vggt, vision_hidden)
        aux_tokens, aux_mask = self.direct_vggt_adapter(
            teacher_current_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs_aug = prefix_embs_aug.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks_aug, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_aug, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs_aug, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_aug, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs_aug,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        flow_loss = F.mse_loss(u_t, v_t)
        zero = flow_loss.detach() * 0.0

        return {
            "loss": flow_loss,
            "flow_loss": flow_loss,
            "base_flow_loss": flow_loss.detach(),
            "current_geo_loss": zero,
            "direct_vggt_aux_abs_mean": aux_tokens.float().abs().mean().detach(),
            "direct_vggt_tokens": torch.tensor(
                teacher_current_hidden.shape[1], dtype=torch.float32, device=flow_loss.device
            ),
            "direct_vggt_gate_abs_mean": self.direct_vggt_adapter.gate.float().abs().mean().detach(),
            "direct_vggt_gate_max_abs": self.direct_vggt_adapter.gate.float().abs().max().detach(),
            "scheme2_aux_tokens": torch.tensor(aux_tokens.shape[1], dtype=torch.float32, device=flow_loss.device),
            "scheme2_injection_is_concat": torch.tensor(
                float(self.scheme2_injection_mode == "concat"), dtype=torch.float32, device=flow_loss.device
            ),
        }

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        device = torch.device(device)
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :].detach()
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        vggt_runtime = self._get_direct_vggt_runtime(device)
        teacher_current_hidden = self._direct_vggt_tokens(images, vggt_runtime, vision_hidden)
        aux_tokens, aux_mask = self.direct_vggt_adapter(
            teacher_current_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs_aug,
            prefix_pad_masks_aug,
            prefix_att_masks_aug,
            use_cache=True,
            output_hidden_states=False,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks_aug, past_key_values, x_t, time.expand(bsize), None)
            x_t = x_t + dt * v_t
            time += dt
        return x_t


class Scheme3D3FusionPI0Pytorch(Scheme2VisualConcatPI0Pytorch):
    """Scheme3-D3: current-geometry plus estimated-action delta fusion.

    The model keeps Scheme2's current-geometry residual-prefix path, then adds a
    suffix-side zero-gated fusion module.  At each flow step it first computes
    the ordinary action velocity, estimates the denoised endpoint
    ``action_hat = x_t - t * v_t`` with stop-gradient, predicts compact
    delta-shaped tokens from predicted current geometry plus ``action_hat``,
    and fuses those tokens into the action suffix states.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        scheme3_hidden_dim_raw = getattr(extra_config, "scheme3_plan_hidden_dim", None)
        scheme3_hidden_dim = proposal_hidden_dim if scheme3_hidden_dim_raw is None else int(scheme3_hidden_dim_raw)
        scheme3_horizon_raw = getattr(extra_config, "scheme3_horizon_id", None)
        if scheme3_horizon_raw is None:
            data_config = getattr(extra_config, "data", None)
            scheme3_horizon_raw = getattr(data_config, "future_teacher_obs_offset", None)
        scheme3_horizon_id = None if scheme3_horizon_raw is None else int(scheme3_horizon_raw)
        scheme3_max_horizon = max(64, 1 if scheme3_horizon_id is None else scheme3_horizon_id + 1)

        self.scheme3_d3_adapter = Scheme3D3ActionDeltaAdapter(
            current_dim=self.teacher_dim,
            suffix_dim=self.action_out_proj.in_features,
            delta_dim=int(getattr(extra_config, "scheme3_delta_dim", 256)),
            num_delta_tokens=int(getattr(extra_config, "scheme3_num_delta_tokens", 8)),
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            state_dim=self.config.action_dim,
            hidden_dim=scheme3_hidden_dim,
            num_layers=int(getattr(extra_config, "scheme3_plan_num_layers", 3)),
            num_heads=self.geo_num_heads,
            max_horizon=scheme3_max_horizon,
            horizon_id=scheme3_horizon_id,
            dropout=proposal_dropout,
        )

    def _scheme3_apply_delta_fusion(
        self,
        *,
        suffix_out: torch.Tensor,
        current_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        base_v_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_hat = estimate_action_endpoint(x_t.float(), timestep.float(), base_v_t.float(), detach=True)
        refined_suffix, pred_delta_tokens = self.scheme3_d3_adapter(
            suffix_out,
            current_tokens,
            action_hat,
            state=state.float(),
            token_mask=vision_mask,
        )
        v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), refined_suffix)
        return v_t, pred_delta_tokens, action_hat

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align]
            vision_hidden = prefix_hidden[:, :img_len, :].detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        current_geo_loss, spatial_metrics = self._scheme2_spatial_loss(
            pred_current_hidden, teacher_current_hidden, vision_mask
        )

        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs_aug = prefix_embs_aug.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks_aug, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_aug, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs_aug, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_aug, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs_aug,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        base_v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        v_t, pred_delta_tokens, action_hat = self._scheme3_apply_delta_fusion(
            suffix_out=suffix_out,
            current_tokens=pred_current_hidden,
            vision_mask=vision_mask,
            state=state,
            x_t=x_t,
            timestep=time,
            base_v_t=base_v_t,
        )
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "scheme2_aux_abs_mean": aux_tokens.float().abs().mean().detach(),
            "scheme2_aux_tokens": torch.tensor(aux_tokens.shape[1], dtype=torch.float32, device=loss.device),
            "scheme2_injection_is_concat": torch.tensor(
                float(self.scheme2_injection_mode == "concat"), dtype=torch.float32, device=loss.device
            ),
            "scheme3_d3_delta_abs_mean": pred_delta_tokens.float().abs().mean().detach(),
            "scheme3_d3_action_hat_abs_mean": action_hat.float().abs().mean().detach(),
            "scheme3_d3_gate_abs_mean": self.scheme3_d3_adapter.gate_abs_mean().detach(),
            "scheme3_d3_gate_max_abs": self.scheme3_d3_adapter.gate_max_abs().detach(),
        }
        output.update(spatial_metrics)
        output.update(self._scheme2_extra_metrics())
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align]
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs_aug,
            prefix_pad_masks_aug,
            prefix_att_masks_aug,
            use_cache=True,
            output_hidden_states=False,
        )
        geo_cache = {
            "current_tokens": pred_current_hidden.float(),
            "vision_mask": None if vision_mask is None else vision_mask.bool(),
        }

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks_aug, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache=None):
        if geo_cache is None or "current_tokens" not in geo_cache:
            raise ValueError("Scheme3-D3 denoise_step requires geo_cache with current_tokens")
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        base_v_t = self.action_out_proj(suffix_out)
        v_t, _pred_delta_tokens, _action_hat = self._scheme3_apply_delta_fusion(
            suffix_out=suffix_out,
            current_tokens=geo_cache["current_tokens"],
            vision_mask=geo_cache.get("vision_mask"),
            state=state,
            x_t=x_t,
            timestep=timestep,
            base_v_t=base_v_t,
        )
        return v_t


class Scheme3CorrectedDeltaPI0Pytorch(Scheme2VisualConcatPI0Pytorch):
    """Corrected Delta Align v1.

    This is the first policy-ready Delta-only branch under the revised Double
    Align plan.  It uses frozen VGGT and a frozen DeltaTokenizer only for
    training targets.  At inference it predicts compact delta tokens from
    deployable pi visual/language/state features and injects them through a
    zero-gated suffix fusion path.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        scheme3_hidden_dim_raw = getattr(extra_config, "scheme3_plan_hidden_dim", None)
        scheme3_hidden_dim = proposal_hidden_dim if scheme3_hidden_dim_raw is None else int(scheme3_hidden_dim_raw)
        scheme3_horizon_raw = getattr(extra_config, "scheme3_horizon_id", None)
        if scheme3_horizon_raw is None:
            data_config = getattr(extra_config, "data", None)
            scheme3_horizon_raw = getattr(data_config, "future_teacher_obs_offset", None)
        self.scheme3_horizon_id = None if scheme3_horizon_raw is None else int(scheme3_horizon_raw)

        self.scheme3_delta_token_loss_coeff = float(getattr(extra_config, "scheme3_delta_token_loss_coeff", 1.0))
        self.scheme3_delta_future_loss_coeff = float(getattr(extra_config, "scheme3_delta_future_loss_coeff", 1.0))
        self.scheme3_tokenizer = self._load_scheme3_tokenizer(
            getattr(extra_config, "scheme3_tokenizer_checkpoint", None)
        )
        tokenizer_config = self.scheme3_tokenizer.config
        if int(getattr(extra_config, "scheme3_delta_dim", tokenizer_config.delta_dim)) != tokenizer_config.delta_dim:
            raise ValueError(
                "scheme3_delta_dim must match frozen tokenizer delta_dim="
                f"{tokenizer_config.delta_dim}, got {getattr(extra_config, 'scheme3_delta_dim', None)}"
            )
        if int(getattr(extra_config, "scheme3_num_delta_tokens", tokenizer_config.num_delta_tokens)) != (
            tokenizer_config.num_delta_tokens
        ):
            raise ValueError(
                "scheme3_num_delta_tokens must match frozen tokenizer num_delta_tokens="
                f"{tokenizer_config.num_delta_tokens}, got {getattr(extra_config, 'scheme3_num_delta_tokens', None)}"
            )
        if tokenizer_config.teacher_dim != self.teacher_dim:
            raise ValueError(
                f"frozen tokenizer teacher_dim={tokenizer_config.teacher_dim} must match model teacher_dim={self.teacher_dim}"
            )
        if self.scheme3_horizon_id is not None and self.scheme3_horizon_id >= tokenizer_config.max_horizon:
            raise ValueError(
                f"scheme3_horizon_id={self.scheme3_horizon_id} exceeds tokenizer max_horizon={tokenizer_config.max_horizon}"
            )

        self.scheme3_current_adapter = Scheme3DeployableCurrentAdapter(
            visual_dim=self.LLM_width,
            teacher_dim=self.teacher_dim,
            hidden_dim=proposal_hidden_dim,
            dropout=proposal_dropout,
        )
        self.scheme3_delta_adapter = Scheme3CorrectedDeltaAdapter(
            source_dim=self.teacher_dim,
            language_dim=self.LLM_width,
            state_dim=self.config.action_dim,
            suffix_dim=self.action_out_proj.in_features,
            hidden_dim=scheme3_hidden_dim,
            delta_dim=tokenizer_config.delta_dim,
            num_delta_tokens=tokenizer_config.num_delta_tokens,
            num_layers=int(getattr(extra_config, "scheme3_plan_num_layers", 3)),
            num_heads=self.geo_num_heads,
            max_horizon=tokenizer_config.max_horizon,
            horizon_id=self.scheme3_horizon_id,
            dropout=proposal_dropout,
        )
        _set_module_trainable(self.scheme3_tokenizer, False)
        self.scheme3_tokenizer.eval()

        # This variant is Delta-only: do not train or use Scheme2 prefix injection.
        _set_module_trainable(self.scheme2_adapter, False)

    def _load_scheme3_tokenizer(self, checkpoint_path: str | None) -> DeltaTokenizer:
        if checkpoint_path is None:
            raise ValueError("scheme3_delta requires extra_config.scheme3_tokenizer_checkpoint")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DeltaTokenizer checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
            raise ValueError(f"Invalid DeltaTokenizer checkpoint payload at {checkpoint_path}")
        tokenizer = DeltaTokenizer(DeltaTokenizerConfig(**payload["config"]))
        tokenizer.load_state_dict(payload["state_dict"])
        tokenizer.eval()
        return tokenizer

    def _scheme3_horizon_ids(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.scheme3_horizon_id is None:
            return None
        return torch.full((batch_size,), self.scheme3_horizon_id, dtype=torch.long, device=device)

    def _language_summary(
        self,
        prefix_hidden: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        img_len: int,
    ) -> torch.Tensor:
        if prefix_hidden.ndim != 3:
            raise ValueError(f"prefix_hidden must be rank 3, got shape {tuple(prefix_hidden.shape)}")
        if prefix_pad_masks.shape != prefix_hidden.shape[:2]:
            raise ValueError(
                f"prefix_pad_masks shape {tuple(prefix_pad_masks.shape)} must match prefix_hidden "
                f"{tuple(prefix_hidden.shape[:2])}"
            )
        if img_len >= prefix_hidden.shape[1]:
            return torch.zeros(prefix_hidden.shape[0], prefix_hidden.shape[-1], device=prefix_hidden.device)
        lang_hidden = prefix_hidden[:, img_len:, :].float()
        lang_mask = prefix_pad_masks[:, img_len:].to(dtype=lang_hidden.dtype, device=lang_hidden.device).unsqueeze(-1)
        denom = lang_mask.sum(dim=1).clamp_min(1.0)
        return (lang_hidden * lang_mask).sum(dim=1) / denom

    def _delta_token_loss(
        self,
        pred_delta_tokens: torch.Tensor,
        target_delta_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred_delta_tokens.shape != target_delta_tokens.shape:
            raise ValueError(
                f"pred_delta_tokens shape {tuple(pred_delta_tokens.shape)} must match target "
                f"{tuple(target_delta_tokens.shape)}"
            )
        smooth_l1 = F.smooth_l1_loss(pred_delta_tokens.float(), target_delta_tokens.float())
        pred_norm = F.normalize(pred_delta_tokens.float(), dim=-1)
        target_norm = F.normalize(target_delta_tokens.float(), dim=-1)
        cosine = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
        loss = smooth_l1 + 0.1 * cosine
        return loss, {
            "scheme3_delta_token_smooth_l1": smooth_l1.detach(),
            "scheme3_delta_token_cosine": cosine.detach(),
        }

    def _future_loss(
        self,
        pred_future_tokens: torch.Tensor,
        target_future_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = geometry_distance(
            pred_future_tokens,
            target_future_tokens.detach(),
            vision_mask,
            relation_max_tokens=self.scheme3_tokenizer.config.relation_max_tokens,
        )
        return loss, {f"scheme3_future_{key}": value.detach() for key, value in metrics.items()}

    def _teacher_delta_targets(
        self,
        teacher_current_hidden: torch.Tensor,
        teacher_future_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_ids = self._scheme3_horizon_ids(teacher_current_hidden.shape[0], teacher_current_hidden.device)
        with torch.no_grad():
            target_delta_tokens = self.scheme3_tokenizer.encode(
                teacher_current_hidden.detach(),
                teacher_future_hidden.detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
            oracle_future_recon = self.scheme3_tokenizer.decode(
                teacher_current_hidden.detach(),
                target_delta_tokens.detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
        return target_delta_tokens.detach(), oracle_future_recon.detach()

    def _predicted_delta_and_future(
        self,
        current_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_delta_tokens = self.scheme3_delta_adapter.predictor(
            current_tokens,
            token_mask=vision_mask,
            language_features=language_features,
            state=state.float(),
            action_intent=action_intent,
        )
        horizon_ids = self._scheme3_horizon_ids(current_tokens.shape[0], current_tokens.device)
        pred_future_tokens = self.scheme3_tokenizer.decode(
            current_tokens,
            pred_delta_tokens,
            token_mask=vision_mask,
            horizon_ids=horizon_ids,
        )
        return pred_delta_tokens, pred_future_tokens

    def _apply_scheme3_delta_fusion(
        self,
        suffix_out: torch.Tensor,
        pred_delta_tokens: torch.Tensor,
    ) -> torch.Tensor:
        refined_suffix = self.scheme3_delta_adapter.suffix_fusion(suffix_out, pred_delta_tokens)
        return self._apply_checkpoint(lambda x: self.action_out_proj(x), refined_suffix)

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            future_teacher_images,
            future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
            vision_hidden = prefix_hidden[:, :img_len, :].detach()
            language_features = self._language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden = self.scheme3_current_adapter(vision_hidden, vision_mask)
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)
        current_geo_loss, spatial_metrics = _spatial_alignment_loss(
            pred_current_hidden,
            teacher_current_hidden.detach(),
            vision_mask,
        )
        target_delta_tokens, oracle_future_recon = self._teacher_delta_targets(
            teacher_current_hidden,
            teacher_future_hidden,
            vision_mask,
        )
        pred_delta_tokens, pred_future_tokens = self._predicted_delta_and_future(
            pred_current_hidden,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
        )
        delta_token_loss, delta_token_metrics = self._delta_token_loss(pred_delta_tokens, target_delta_tokens)
        delta_future_loss, delta_future_metrics = self._future_loss(
            pred_future_tokens,
            teacher_future_hidden,
            vision_mask,
        )
        with torch.no_grad():
            current_as_future_loss, _ = self._future_loss(
                pred_current_hidden.detach(), teacher_future_hidden, vision_mask
            )
            oracle_future_loss, _ = self._future_loss(oracle_future_recon, teacher_future_hidden, vision_mask)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        base_v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        v_t = self._apply_scheme3_delta_fusion(suffix_out, pred_delta_tokens)
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()
        weighted_delta_loss = (
            self.scheme3_delta_token_loss_coeff * delta_token_loss
            + self.scheme3_delta_future_loss_coeff * delta_future_loss
        )
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss + self.delta_loss_coeff * weighted_delta_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "delta_geo_loss": weighted_delta_loss,
            "scheme3_delta_token_loss": delta_token_loss,
            "scheme3_delta_future_loss": delta_future_loss,
            "scheme3_delta_current_as_future_loss": current_as_future_loss.detach(),
            "scheme3_delta_oracle_future_loss": oracle_future_loss.detach(),
            "scheme3_delta_abs_mean": pred_delta_tokens.float().abs().mean().detach(),
            "scheme3_delta_gate_abs_mean": self.scheme3_delta_adapter.gate_abs_mean().detach(),
            "scheme3_delta_gate_max_abs": self.scheme3_delta_adapter.gate_max_abs().detach(),
            "scheme3_delta_future_fallback_ratio": future_teacher_fallback.float().mean().detach()
            if future_teacher_fallback is not None
            else torch.tensor(0.0, dtype=torch.float32, device=loss.device),
            "current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
            "delta_loss_coeff": torch.tensor(self.delta_loss_coeff, device=loss.device),
        }
        output.update(spatial_metrics)
        output.update(delta_token_metrics)
        output.update(delta_future_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
        vision_hidden = prefix_hidden[:, :img_len, :].detach()
        language_features = self._language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden = self.scheme3_current_adapter(vision_hidden, vision_mask)
        pred_delta_tokens, _pred_future_tokens = self._predicted_delta_and_future(
            pred_current_hidden,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
        )
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=True,
            output_hidden_states=False,
        )
        geo_cache = {"pred_delta_tokens": pred_delta_tokens.float()}

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def _scheme3_delta_suffix_hidden_from_cache(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        expected = (batch_size, self.config.action_horizon, self.action_out_proj.in_features)
        if suffix_out.shape != expected:
            raise RuntimeError(f"suffix_out shape {tuple(suffix_out.shape)} does not match {expected}")
        return suffix_out

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache=None):
        if geo_cache is None or "pred_delta_tokens" not in geo_cache:
            raise ValueError("scheme3_delta denoise_step requires geo_cache with pred_delta_tokens")
        suffix_out = self._scheme3_delta_suffix_hidden_from_cache(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        )
        return self._apply_scheme3_delta_fusion(suffix_out, geo_cache["pred_delta_tokens"])


class Scheme3CorrectedDeltaActionIntentPI0Pytorch(Scheme3CorrectedDeltaPI0Pytorch):
    """Corrected Delta-only v2 with deployable action-intent conditioning.

    Compared with ``scheme3_delta``, this variant keeps the same frozen
    DeltaTokenizer target space and teacher-free inference path, but conditions
    delta prediction on a stop-gradient summary of the frozen action expert's
    suffix hidden state.  It does not use GT actions or future observations at
    inference.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        scheme3_hidden_dim_raw = getattr(extra_config, "scheme3_plan_hidden_dim", None)
        scheme3_hidden_dim = proposal_hidden_dim if scheme3_hidden_dim_raw is None else int(scheme3_hidden_dim_raw)
        tokenizer_config = self.scheme3_tokenizer.config
        self.scheme3_delta_adapter = Scheme3CorrectedDeltaAdapter(
            source_dim=self.teacher_dim,
            language_dim=self.LLM_width,
            state_dim=self.config.action_dim,
            suffix_dim=self.action_out_proj.in_features,
            hidden_dim=scheme3_hidden_dim,
            delta_dim=tokenizer_config.delta_dim,
            num_delta_tokens=tokenizer_config.num_delta_tokens,
            num_layers=int(getattr(extra_config, "scheme3_plan_num_layers", 3)),
            num_heads=self.geo_num_heads,
            max_horizon=tokenizer_config.max_horizon,
            horizon_id=self.scheme3_horizon_id,
            action_intent_dim=self.action_out_proj.in_features,
            dropout=proposal_dropout,
        )

    def _action_intent_features(self, suffix_out: torch.Tensor) -> torch.Tensor:
        if suffix_out.ndim != 3:
            raise ValueError(f"suffix_out must be rank 3, got shape {tuple(suffix_out.shape)}")
        expected_dim = self.action_out_proj.in_features
        if suffix_out.shape[-1] != expected_dim:
            raise ValueError(f"suffix_out last dim {suffix_out.shape[-1]} must match {expected_dim}")
        # [B, H_action, D_suffix] -> [B, D_suffix]. Stop-gradient keeps this
        # condition diagnostic/deployable without pushing delta losses into the
        # frozen action expert hidden path.
        return suffix_out.float().mean(dim=1).detach()

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            future_teacher_images,
            future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
            vision_hidden = prefix_hidden[:, :img_len, :].detach()
            language_features = self._language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden = self.scheme3_current_adapter(vision_hidden, vision_mask)
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)
        current_geo_loss, spatial_metrics = _spatial_alignment_loss(
            pred_current_hidden,
            teacher_current_hidden.detach(),
            vision_mask,
        )
        target_delta_tokens, oracle_future_recon = self._teacher_delta_targets(
            teacher_current_hidden,
            teacher_future_hidden,
            vision_mask,
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        action_intent = self._action_intent_features(suffix_out)
        pred_delta_tokens, pred_future_tokens = self._predicted_delta_and_future(
            pred_current_hidden,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        delta_token_loss, delta_token_metrics = self._delta_token_loss(pred_delta_tokens, target_delta_tokens)
        delta_future_loss, delta_future_metrics = self._future_loss(
            pred_future_tokens,
            teacher_future_hidden,
            vision_mask,
        )
        with torch.no_grad():
            current_as_future_loss, _ = self._future_loss(
                pred_current_hidden.detach(), teacher_future_hidden, vision_mask
            )
            oracle_future_loss, _ = self._future_loss(oracle_future_recon, teacher_future_hidden, vision_mask)

        base_v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        v_t = self._apply_scheme3_delta_fusion(suffix_out, pred_delta_tokens)
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()
        weighted_delta_loss = (
            self.scheme3_delta_token_loss_coeff * delta_token_loss
            + self.scheme3_delta_future_loss_coeff * delta_future_loss
        )
        loss = flow_loss + self.current_geo_loss_coeff * current_geo_loss + self.delta_loss_coeff * weighted_delta_loss

        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "delta_geo_loss": weighted_delta_loss,
            "scheme3_delta_token_loss": delta_token_loss,
            "scheme3_delta_future_loss": delta_future_loss,
            "scheme3_delta_current_as_future_loss": current_as_future_loss.detach(),
            "scheme3_delta_oracle_future_loss": oracle_future_loss.detach(),
            "scheme3_delta_abs_mean": pred_delta_tokens.float().abs().mean().detach(),
            "scheme3_delta_action_intent_abs_mean": action_intent.float().abs().mean().detach(),
            "scheme3_delta_gate_abs_mean": self.scheme3_delta_adapter.gate_abs_mean().detach(),
            "scheme3_delta_gate_max_abs": self.scheme3_delta_adapter.gate_max_abs().detach(),
            "scheme3_delta_future_fallback_ratio": future_teacher_fallback.float().mean().detach()
            if future_teacher_fallback is not None
            else torch.tensor(0.0, dtype=torch.float32, device=loss.device),
            "current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
            "delta_loss_coeff": torch.tensor(self.delta_loss_coeff, device=loss.device),
        }
        output.update(spatial_metrics)
        output.update(delta_token_metrics)
        output.update(delta_future_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
        vision_hidden = prefix_hidden[:, :img_len, :].detach()
        language_features = self._language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        pred_current_hidden = self.scheme3_current_adapter(vision_hidden, vision_mask)
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=True,
            output_hidden_states=False,
        )

        initial_time = torch.ones(bsize, dtype=torch.float32, device=device)
        suffix_out = self._scheme3_delta_suffix_hidden_from_cache(
            state,
            prefix_pad_masks,
            past_key_values,
            noise,
            initial_time,
        )
        action_intent = self._action_intent_features(suffix_out)
        pred_delta_tokens, _pred_future_tokens = self._predicted_delta_and_future(
            pred_current_hidden,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        geo_cache = {"pred_delta_tokens": pred_delta_tokens.float()}

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        return x_t


class Scheme4LACDoubleAlignPI0Pytorch(Scheme2UnifiedSpatialPI0Pytorch):
    """Scheme4 Language-Action Conditioned Double Align.

    This variant keeps corrected Scheme2's deployable spatial bottleneck
    ``S_t`` and adds a teacher-free-at-inference delta branch conditioned on
    ``S_t``, masked language summary, frozen action-expert intent, robot state,
    and horizon embedding.
    """

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        proposal_hidden_dim_raw = getattr(extra_config, "proposal_hidden_dim", None)
        proposal_hidden_dim = (
            self.geo_hidden_dim * 2 if proposal_hidden_dim_raw is None else int(proposal_hidden_dim_raw)
        )
        proposal_dropout_raw = getattr(extra_config, "proposal_dropout", None)
        proposal_dropout = self.geo_dropout if proposal_dropout_raw is None else float(proposal_dropout_raw)
        scheme4_hidden_dim_raw = getattr(extra_config, "scheme4_plan_hidden_dim", None)
        if scheme4_hidden_dim_raw is None:
            scheme4_hidden_dim_raw = getattr(extra_config, "scheme3_plan_hidden_dim", None)
        scheme4_hidden_dim = proposal_hidden_dim if scheme4_hidden_dim_raw is None else int(scheme4_hidden_dim_raw)

        scheme4_horizon_raw = getattr(extra_config, "scheme4_horizon_id", None)
        if scheme4_horizon_raw is None:
            scheme4_horizon_raw = getattr(extra_config, "scheme3_horizon_id", None)
        if scheme4_horizon_raw is None:
            data_config = getattr(extra_config, "data", None)
            scheme4_horizon_raw = getattr(data_config, "future_teacher_obs_offset", None)
        self.scheme4_horizon_id = None if scheme4_horizon_raw is None else int(scheme4_horizon_raw)

        self.scheme4_delta_enabled = bool(getattr(extra_config, "scheme4_delta_enabled", True))
        self.scheme4_delta_token_loss_coeff = float(getattr(extra_config, "scheme4_delta_token_loss_coeff", 0.0))
        self.scheme4_future_recon_loss_coeff = float(getattr(extra_config, "scheme4_future_recon_loss_coeff", 0.01))
        self.scheme4_action_intent_mode = str(getattr(extra_config, "scheme4_action_intent_mode", "t1_suffix_mean"))
        self.scheme4_training_intent_timestep = str(
            getattr(extra_config, "scheme4_training_intent_timestep", "sampled_flow")
        )
        if self.scheme4_training_intent_timestep not in {"sampled_flow", "t1_noise"}:
            raise ValueError(
                "scheme4_training_intent_timestep must be 'sampled_flow' or 't1_noise', got "
                f"{self.scheme4_training_intent_timestep!r}"
            )
        self.scheme4_delta_fusion_type = str(getattr(extra_config, "scheme4_delta_fusion_type", "suffix_cross_attn"))
        self.scheme4_use_decoder_at_inference = bool(getattr(extra_config, "scheme4_use_decoder_at_inference", False))
        self.scheme4_inference_condition_ablation = str(
            getattr(extra_config, "scheme4_inference_condition_ablation", "none")
        )
        self.scheme4_inference_delta_control = str(getattr(extra_config, "scheme4_inference_delta_control", "normal"))
        self.scheme4_inference_delta_mean_path = getattr(extra_config, "scheme4_inference_delta_mean_path", None)
        self.scheme4_inference_control_seed = int(getattr(extra_config, "scheme4_inference_control_seed", 0))
        self.scheme4_inference_fusion_scale = float(getattr(extra_config, "scheme4_inference_fusion_scale", 1.0))
        if not math.isfinite(self.scheme4_inference_fusion_scale) or self.scheme4_inference_fusion_scale < 0.0:
            raise ValueError(
                "scheme4_inference_fusion_scale must be finite and non-negative, got "
                f"{self.scheme4_inference_fusion_scale}"
            )
        self.scheme4_adaptive_lookahead_enabled = bool(
            getattr(extra_config, "scheme4_adaptive_lookahead_enabled", False)
        )
        self.scheme4_router_only_training = bool(getattr(extra_config, "scheme4_router_only_training", False))
        if self.scheme4_router_only_training and not self.scheme4_adaptive_lookahead_enabled:
            raise ValueError("scheme4_router_only_training requires scheme4_adaptive_lookahead_enabled=True")
        self.scheme4_inference_lookahead_mode = str(getattr(extra_config, "scheme4_inference_lookahead_mode", "fixed"))
        if self.scheme4_inference_lookahead_mode not in LOOKAHEAD_ROUTING_MODES:
            raise ValueError(
                "scheme4_inference_lookahead_mode must be one of "
                f"{LOOKAHEAD_ROUTING_MODES}, got {self.scheme4_inference_lookahead_mode!r}"
            )
        if self.scheme4_inference_lookahead_mode in {"soft", "hard"} and not (self.scheme4_adaptive_lookahead_enabled):
            raise ValueError("soft/hard lookahead routing requires scheme4_adaptive_lookahead_enabled=True")
        self.scheme4_utility_loss_coeff = float(getattr(extra_config, "scheme4_utility_loss_coeff", 0.1))
        self.scheme4_utility_temperature = float(getattr(extra_config, "scheme4_utility_temperature", 0.05))
        self.scheme4_utility_margin = float(getattr(extra_config, "scheme4_utility_margin", 0.0))
        if not math.isfinite(self.scheme4_utility_loss_coeff) or self.scheme4_utility_loss_coeff < 0.0:
            raise ValueError("scheme4_utility_loss_coeff must be finite and non-negative")
        if not math.isfinite(self.scheme4_utility_temperature) or self.scheme4_utility_temperature <= 0.0:
            raise ValueError("scheme4_utility_temperature must be finite and positive")
        if not math.isfinite(self.scheme4_utility_margin):
            raise ValueError("scheme4_utility_margin must be finite")
        if self.scheme4_adaptive_lookahead_enabled and not self.scheme4_delta_enabled:
            raise ValueError("adaptive lookahead requires scheme4_delta_enabled=True")
        self.scheme4_inference_diagnostics_enabled = bool(
            getattr(extra_config, "scheme4_inference_diagnostics_enabled", False)
        )
        self._scheme4_inference_diagnostic_steps: list[dict[str, torch.Tensor]] | None = None
        self._scheme4_latest_inference_diagnostics: dict[str, torch.Tensor] | None = None
        self._scheme4_inference_control_step = 0
        self._scheme4_prev_language_features: torch.Tensor | None = None
        self._scheme4_prev_action_intent: torch.Tensor | None = None
        self._scheme4_prev_delta_tokens: torch.Tensor | None = None
        self.scheme4_condition_negative_loss_coeff = float(
            getattr(extra_config, "scheme4_condition_negative_loss_coeff", 0.0)
        )
        self.scheme4_condition_negative_margin = float(getattr(extra_config, "scheme4_condition_negative_margin", 0.05))
        negative_modes_raw = str(getattr(extra_config, "scheme4_condition_negative_modes", "language,action"))
        self.scheme4_condition_negative_modes = tuple(
            mode.strip() for mode in negative_modes_raw.split(",") if mode.strip()
        )
        invalid_negative_modes = set(self.scheme4_condition_negative_modes) - {"language", "action"}
        if invalid_negative_modes:
            raise ValueError(
                "scheme4_condition_negative_modes only supports 'language' and 'action', got "
                f"{sorted(invalid_negative_modes)}"
            )
        self.scheme4_training_action_condition = str(
            getattr(extra_config, "scheme4_training_action_condition", "normal")
        )
        if self.scheme4_training_action_condition not in SCHEME4_TRAINING_ACTION_CONDITIONS:
            raise ValueError(
                "scheme4_training_action_condition must be one of "
                f"{SCHEME4_TRAINING_ACTION_CONDITIONS}, got {self.scheme4_training_action_condition!r}"
            )
        if (
            self.scheme4_training_action_condition == "zero_action"
            and "action" in self.scheme4_condition_negative_modes
        ):
            raise ValueError(
                "scheme4_training_action_condition='zero_action' requires a condition-negative objective "
                "without the action mode because zero action conditions cannot be meaningfully shuffled"
            )
        if self.scheme4_use_decoder_at_inference:
            raise ValueError(
                "scheme4_use_decoder_at_inference=True is not supported; scheme4 inference is decoder-free"
            )
        if self.scheme4_inference_condition_ablation not in SCHEME4_INFERENCE_CONDITION_ABLATIONS:
            raise ValueError(
                "scheme4_inference_condition_ablation must be one of "
                f"{SCHEME4_INFERENCE_CONDITION_ABLATIONS}, got {self.scheme4_inference_condition_ablation!r}"
            )
        if self.scheme4_inference_delta_control not in SCHEME4_INFERENCE_DELTA_CONTROLS:
            raise ValueError(
                "scheme4_inference_delta_control must be one of "
                f"{SCHEME4_INFERENCE_DELTA_CONTROLS}, got {self.scheme4_inference_delta_control!r}"
            )
        if not self.scheme4_delta_enabled and (
            self.scheme4_delta_token_loss_coeff != 0.0 or self.scheme4_future_recon_loss_coeff != 0.0
        ):
            raise ValueError(
                "scheme4_delta_enabled=False requires scheme4_delta_token_loss_coeff=0.0 and "
                "scheme4_future_recon_loss_coeff=0.0"
            )

        self.scheme4_tokenizer = self._load_scheme4_tokenizer(
            getattr(extra_config, "scheme4_tokenizer_checkpoint", None)
        )
        tokenizer_config = self.scheme4_tokenizer.config
        if int(getattr(extra_config, "scheme4_delta_dim", tokenizer_config.delta_dim)) != tokenizer_config.delta_dim:
            raise ValueError(
                "scheme4_delta_dim must match frozen tokenizer delta_dim="
                f"{tokenizer_config.delta_dim}, got {getattr(extra_config, 'scheme4_delta_dim', None)}"
            )
        if int(getattr(extra_config, "scheme4_num_delta_tokens", tokenizer_config.num_delta_tokens)) != (
            tokenizer_config.num_delta_tokens
        ):
            raise ValueError(
                "scheme4_num_delta_tokens must match frozen tokenizer num_delta_tokens="
                f"{tokenizer_config.num_delta_tokens}, got {getattr(extra_config, 'scheme4_num_delta_tokens', None)}"
            )
        if tokenizer_config.teacher_dim != self.teacher_dim:
            raise ValueError(
                f"frozen tokenizer teacher_dim={tokenizer_config.teacher_dim} must match model teacher_dim={self.teacher_dim}"
            )
        if self.scheme4_horizon_id is not None and self.scheme4_horizon_id >= tokenizer_config.max_horizon:
            raise ValueError(
                f"scheme4_horizon_id={self.scheme4_horizon_id} exceeds tokenizer max_horizon={tokenizer_config.max_horizon}"
            )
        mean_delta_tokens = self._load_scheme4_inference_delta_mean(
            self.scheme4_inference_delta_mean_path,
            num_delta_tokens=tokenizer_config.num_delta_tokens,
            delta_dim=tokenizer_config.delta_dim,
        )
        self.register_buffer("_scheme4_inference_delta_mean", mean_delta_tokens, persistent=False)

        self.scheme4_delta_adapter = Scheme4LanguageActionDeltaAdapter(
            spatial_dim=self.teacher_dim,
            language_dim=self.LLM_width,
            state_dim=self.config.action_dim,
            action_intent_dim=self.action_out_proj.in_features,
            suffix_dim=self.action_out_proj.in_features,
            hidden_dim=scheme4_hidden_dim,
            delta_dim=tokenizer_config.delta_dim,
            num_delta_tokens=tokenizer_config.num_delta_tokens,
            num_layers=int(getattr(extra_config, "scheme4_plan_num_layers", 3)),
            num_heads=self.geo_num_heads,
            max_horizon=tokenizer_config.max_horizon,
            horizon_id=self.scheme4_horizon_id,
            action_intent_mode=self.scheme4_action_intent_mode,
            delta_fusion_type=self.scheme4_delta_fusion_type,
            delta_enabled=self.scheme4_delta_enabled,
            dropout=proposal_dropout,
        )
        if self.scheme4_adaptive_lookahead_enabled:
            self.scheme4_utility_router: Scheme4UtilityRouter | None = Scheme4UtilityRouter(
                spatial_dim=self.teacher_dim,
                language_dim=self.LLM_width,
                state_dim=self.config.action_dim,
                action_intent_dim=self.action_out_proj.in_features,
                hidden_dim=int(getattr(extra_config, "scheme4_utility_hidden_dim", 128)),
            )
        else:
            # Avoid adding checkpoint keys or compute to legacy fixed Route A.
            self.scheme4_utility_router = None
        _set_module_trainable(self.scheme4_tokenizer, False)
        self.scheme4_tokenizer.eval()

    def _load_scheme4_tokenizer(self, checkpoint_path: str | None) -> DeltaTokenizer:
        if checkpoint_path is None:
            raise ValueError("scheme4_lac_double_align requires extra_config.scheme4_tokenizer_checkpoint")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Scheme4 DeltaTokenizer checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
            raise ValueError(f"Invalid DeltaTokenizer checkpoint payload at {checkpoint_path}")
        tokenizer = DeltaTokenizer(DeltaTokenizerConfig(**payload["config"]))
        tokenizer.load_state_dict(payload["state_dict"])
        tokenizer.eval()
        return tokenizer

    def _load_scheme4_inference_delta_mean(
        self,
        path: str | None,
        *,
        num_delta_tokens: int,
        delta_dim: int,
    ) -> torch.Tensor | None:
        if path is None:
            if self.scheme4_inference_delta_control == "mean_delta":
                raise ValueError(
                    "scheme4_inference_delta_control='mean_delta' requires scheme4_inference_delta_mean_path"
                )
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scheme4 inference mean-delta file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("mean_delta_tokens", "delta_mean", "mean", "tensor"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, torch.Tensor):
            raise ValueError(f"Mean-delta payload at {path} must be a Tensor or dict containing a Tensor")
        tensor = payload.detach().float().cpu()
        if tensor.ndim == 2:
            expected = (int(num_delta_tokens), int(delta_dim))
            if tensor.shape != expected:
                raise ValueError(f"mean-delta tensor shape {tuple(tensor.shape)} must match {expected}")
        elif tensor.ndim == 3:
            expected = (1, int(num_delta_tokens), int(delta_dim))
            if tensor.shape != expected:
                raise ValueError(f"mean-delta tensor shape {tuple(tensor.shape)} must match {expected}")
        else:
            raise ValueError(f"mean-delta tensor must be rank 2 or 3, got shape {tuple(tensor.shape)}")
        return tensor

    def _scheme4_inference_generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.scheme4_inference_control_seed + self._scheme4_inference_control_step)
        self._scheme4_inference_control_step += 1
        return generator

    def _scheme4_apply_inference_condition_control(
        self,
        language_features: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_scheme4_inference_condition_ablation(
            self.scheme4_inference_condition_ablation,
            language_features,
            action_intent,
            previous_language_features=self._scheme4_prev_language_features,
            previous_action_intent=self._scheme4_prev_action_intent,
        )

    def _scheme4_apply_inference_delta_control(self, pred_delta_tokens: torch.Tensor) -> torch.Tensor:
        generator = None
        if self.scheme4_inference_delta_control in {"random_delta", "norm_matched_random_delta"}:
            generator = self._scheme4_inference_generator(pred_delta_tokens.device)
        return apply_scheme4_inference_delta_control(
            self.scheme4_inference_delta_control,
            pred_delta_tokens,
            mean_delta_tokens=self._scheme4_inference_delta_mean,
            previous_delta_tokens=self._scheme4_prev_delta_tokens,
            generator=generator,
        )

    def _scheme4_update_inference_control_cache(
        self,
        language_features: torch.Tensor,
        action_intent: torch.Tensor,
        pred_delta_tokens: torch.Tensor,
    ) -> None:
        self._scheme4_prev_language_features = language_features.detach().clone()
        self._scheme4_prev_action_intent = action_intent.detach().clone()
        self._scheme4_prev_delta_tokens = pred_delta_tokens.detach().clone()

    def _scheme4_horizon_ids(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.scheme4_horizon_id is None:
            return None
        return torch.full((batch_size,), self.scheme4_horizon_id, dtype=torch.long, device=device)

    def _scheme4_language_summary(
        self,
        prefix_hidden: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        img_len: int,
    ) -> torch.Tensor:
        if prefix_hidden.ndim != 3:
            raise ValueError(f"prefix_hidden must be rank 3, got shape {tuple(prefix_hidden.shape)}")
        if prefix_pad_masks.shape != prefix_hidden.shape[:2]:
            raise ValueError(
                f"prefix_pad_masks shape {tuple(prefix_pad_masks.shape)} must match prefix_hidden "
                f"{tuple(prefix_hidden.shape[:2])}"
            )
        if img_len >= prefix_hidden.shape[1]:
            return torch.zeros(prefix_hidden.shape[0], prefix_hidden.shape[-1], device=prefix_hidden.device)
        lang_hidden = prefix_hidden[:, img_len:, :].float()
        lang_mask = prefix_pad_masks[:, img_len:].to(dtype=lang_hidden.dtype, device=lang_hidden.device).unsqueeze(-1)
        denom = lang_mask.sum(dim=1).clamp_min(1.0)
        return (lang_hidden * lang_mask).sum(dim=1) / denom

    def _scheme4_action_intent_features(self, suffix_out: torch.Tensor) -> torch.Tensor:
        if self.scheme4_action_intent_mode != "t1_suffix_mean":
            raise NotImplementedError(
                f"scheme4 action_intent_mode={self.scheme4_action_intent_mode!r} is not implemented"
            )
        if suffix_out.ndim != 3:
            raise ValueError(f"suffix_out must be rank 3, got shape {tuple(suffix_out.shape)}")
        expected_dim = self.action_out_proj.in_features
        if suffix_out.shape[-1] != expected_dim:
            raise ValueError(f"suffix_out last dim {suffix_out.shape[-1]} must match {expected_dim}")
        return suffix_out.float().mean(dim=1).detach()

    def _scheme4_delta_token_loss(
        self,
        pred_delta_tokens: torch.Tensor,
        target_delta_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred_delta_tokens.shape != target_delta_tokens.shape:
            raise ValueError(
                f"pred_delta_tokens shape {tuple(pred_delta_tokens.shape)} must match target "
                f"{tuple(target_delta_tokens.shape)}"
            )
        smooth_l1 = F.smooth_l1_loss(pred_delta_tokens.float(), target_delta_tokens.float())
        pred_norm = F.normalize(pred_delta_tokens.float(), dim=-1)
        target_norm = F.normalize(target_delta_tokens.float(), dim=-1)
        cosine = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
        loss = smooth_l1 + 0.1 * cosine
        return loss, {
            "scheme4_delta_token_smooth_l1": smooth_l1.detach(),
            "scheme4_delta_token_cosine": cosine.detach(),
        }

    def _scheme4_condition_negative_loss(
        self,
        pred_delta_tokens: torch.Tensor,
        target_delta_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = pred_delta_tokens.float().sum() * 0.0
        if self.scheme4_condition_negative_loss_coeff == 0.0 or not self.scheme4_condition_negative_modes:
            return zero, {}

        normal_distance = delta_token_distance(pred_delta_tokens, target_delta_tokens.detach())
        losses: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {
            "scheme4_condition_normal_delta_distance": normal_distance.detach(),
        }

        if "language" in self.scheme4_condition_negative_modes:
            neg_language, valid = roll_batch_condition(language_features)
            neg_delta = self._scheme4_predict_delta(
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=neg_language,
                state=state,
                action_intent=action_intent,
            )
            neg_distance = delta_token_distance(neg_delta, target_delta_tokens.detach())
            loss = margin_negative_condition_loss(
                normal_distance,
                neg_distance,
                margin=self.scheme4_condition_negative_margin,
                valid=valid,
            )
            losses.append(loss)
            metrics["scheme4_condition_neg_language_loss"] = loss.detach()
            metrics["scheme4_condition_neg_language_delta_distance"] = neg_distance.detach()
            metrics["scheme4_condition_neg_language_valid"] = valid.to(dtype=torch.float32).detach()

        if "action" in self.scheme4_condition_negative_modes:
            neg_action_intent, valid = roll_batch_condition(action_intent)
            neg_delta = self._scheme4_predict_delta(
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=language_features,
                state=state,
                action_intent=neg_action_intent,
            )
            neg_distance = delta_token_distance(neg_delta, target_delta_tokens.detach())
            loss = margin_negative_condition_loss(
                normal_distance,
                neg_distance,
                margin=self.scheme4_condition_negative_margin,
                valid=valid,
            )
            losses.append(loss)
            metrics["scheme4_condition_neg_action_loss"] = loss.detach()
            metrics["scheme4_condition_neg_action_delta_distance"] = neg_distance.detach()
            metrics["scheme4_condition_neg_action_valid"] = valid.to(dtype=torch.float32).detach()

        if not losses:
            return zero, metrics
        total = torch.stack(losses).mean()
        metrics["scheme4_condition_negative_loss"] = total.detach()
        return total, metrics

    def _scheme4_future_loss(
        self,
        pred_future_tokens: torch.Tensor,
        target_future_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = geometry_distance(
            pred_future_tokens,
            target_future_tokens.detach(),
            vision_mask,
            relation_max_tokens=self.scheme4_tokenizer.config.relation_max_tokens,
        )
        return loss, {f"scheme4_future_{key}": value.detach() for key, value in metrics.items()}

    def _scheme4_teacher_delta_targets(
        self,
        teacher_current_hidden: torch.Tensor,
        teacher_future_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_ids = self._scheme4_horizon_ids(teacher_current_hidden.shape[0], teacher_current_hidden.device)
        with torch.no_grad():
            target_delta_tokens = self.scheme4_tokenizer.encode(
                teacher_current_hidden.detach(),
                teacher_future_hidden.detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
            oracle_future_recon = self.scheme4_tokenizer.decode(
                teacher_current_hidden.detach(),
                target_delta_tokens.detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
        return target_delta_tokens.detach(), oracle_future_recon.detach()

    def _scheme4_predict_delta(
        self,
        spatial_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> torch.Tensor:
        return self.scheme4_delta_adapter.predict_delta(
            spatial_tokens,
            token_mask=vision_mask,
            language_features=language_features,
            state=state.float(),
            action_intent=action_intent,
        )

    def _scheme4_decode_future(
        self,
        spatial_tokens: torch.Tensor,
        pred_delta_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon_ids = self._scheme4_horizon_ids(spatial_tokens.shape[0], spatial_tokens.device)
        return self.scheme4_tokenizer.decode(
            spatial_tokens,
            pred_delta_tokens,
            token_mask=vision_mask,
            horizon_ids=horizon_ids,
        )

    def _apply_scheme4_delta_fusion(
        self,
        suffix_out: torch.Tensor,
        spatial_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
        pred_delta_tokens: torch.Tensor | None = None,
        fusion_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred_delta_tokens is None:
            refined_suffix, pred_delta_tokens, delta_residual = self.scheme4_delta_adapter(
                suffix_out,
                spatial_tokens,
                token_mask=vision_mask,
                language_features=language_features,
                state=state.float(),
                action_intent=action_intent,
                residual_scale=fusion_scale,
            )
        else:
            if self.scheme4_delta_enabled:
                refined_suffix = self.scheme4_delta_adapter.suffix_fusion(
                    suffix_out,
                    pred_delta_tokens,
                    residual_scale=fusion_scale,
                )
            else:
                refined_suffix = suffix_out.float()
            delta_residual = refined_suffix.float() - suffix_out.float()
        v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), refined_suffix)
        return v_t, pred_delta_tokens, delta_residual

    def _scheme4_lookahead_probability(
        self,
        spatial_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
        language_features: torch.Tensor,
        state: torch.Tensor,
        action_intent: torch.Tensor,
    ) -> torch.Tensor:
        if self.scheme4_utility_router is None:
            raise RuntimeError("utility router is unavailable because adaptive lookahead is disabled")
        return self.scheme4_utility_router(
            spatial_tokens,
            token_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        (
            images,
            _img_wo_aug,
            teacher_images,
            future_teacher_images,
            future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True, get_wo_aug=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        with torch.no_grad():
            _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
                output_hidden_states=True,
            )
            prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
            vision_hidden = prefix_hidden[:, :img_len, :].detach()
            language_features = self._scheme4_language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()

        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        spatial_tokens, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        current_geo_loss, spatial_metrics, spatial_loss_skipped = self._scheme2_current_geo_loss(
            spatial_tokens,
            teacher_images,
            vggt,
            vision_hidden,
            vision_mask,
        )

        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs_aug = prefix_embs_aug.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks_aug, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_aug, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs_aug, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_aug, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                output_hidden_states=False,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs_aug,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        initial_noise_suffix_out = None
        if self.scheme4_training_intent_timestep == "t1_noise":
            initial_time = torch.ones(actions.shape[0], dtype=time.dtype, device=time.device)
            with torch.no_grad():
                intent_suffix_embs, intent_pad_masks, intent_att_masks, intent_adarms_cond = self.embed_suffix(
                    state, noise, initial_time
                )
                if (
                    self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                    == torch.bfloat16
                ):
                    intent_suffix_embs = intent_suffix_embs.to(dtype=torch.bfloat16)
                intent_pad_masks_all = torch.cat([prefix_pad_masks_aug, intent_pad_masks], dim=1)
                intent_att_masks_all = torch.cat([prefix_att_masks_aug, intent_att_masks], dim=1)
                intent_attention_mask = self._prepare_attention_masks_4d(
                    make_att_2d_masks(intent_pad_masks_all, intent_att_masks_all)
                )
                intent_position_ids = torch.cumsum(intent_pad_masks_all, dim=1) - 1
                intent_suffix_out = forward_func(
                    prefix_embs_aug,
                    intent_suffix_embs,
                    intent_attention_mask,
                    intent_position_ids,
                    intent_adarms_cond,
                )
                initial_noise_suffix_out = intent_suffix_out[:, -self.config.action_horizon :].float()
        intent_suffix = select_scheme4_training_intent_suffix(
            suffix_out,
            initial_noise_suffix_out,
            self.scheme4_training_intent_timestep,
        )
        action_intent = self._scheme4_action_intent_features(intent_suffix)
        action_condition = apply_scheme4_training_action_condition(
            self.scheme4_training_action_condition,
            action_intent,
        )
        lookahead_probability = None
        if self.scheme4_adaptive_lookahead_enabled:
            lookahead_probability = self._scheme4_lookahead_probability(
                spatial_tokens,
                vision_mask,
                language_features,
                state,
                action_condition,
            )
        pred_delta_tokens = self._scheme4_predict_delta(
            spatial_tokens,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_condition,
        )
        base_v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        if lookahead_probability is None:
            v_t, pred_delta_tokens, delta_residual = self._apply_scheme4_delta_fusion(
                suffix_out,
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=language_features,
                state=state,
                action_intent=action_condition,
                pred_delta_tokens=pred_delta_tokens,
            )
        else:
            fusion_scale = training_fusion_scale(
                lookahead_probability,
                router_only=self.scheme4_router_only_training,
            )
            v_t, pred_delta_tokens, delta_residual = self._apply_scheme4_delta_fusion(
                suffix_out,
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=language_features,
                state=state,
                action_intent=action_condition,
                pred_delta_tokens=pred_delta_tokens,
                fusion_scale=fusion_scale,
            )
            full_v_t, _, _ = self._apply_scheme4_delta_fusion(
                suffix_out,
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=language_features,
                state=state,
                action_intent=action_condition,
                pred_delta_tokens=pred_delta_tokens,
            )
            base_per_example = (u_t.float() - base_v_t.float()).square().mean(dim=(1, 2))
            full_per_example = (u_t.float() - full_v_t.float()).square().mean(dim=(1, 2))
            utility_benefit, utility_target = utility_targets_from_losses(
                base_per_example,
                full_per_example,
                temperature=self.scheme4_utility_temperature,
                margin=self.scheme4_utility_margin,
            )
            utility_loss = utility_supervision_loss(lookahead_probability, utility_target)

        if lookahead_probability is None:
            flow_loss = F.mse_loss(u_t, v_t)
        else:
            per_example_flow_loss = (u_t.float() - v_t.float()).square().mean(dim=(1, 2))
            flow_loss = per_example_flow_loss.mean()
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()

        zero = spatial_tokens.float().sum() * 0.0
        delta_token_loss = zero
        future_recon_loss = zero
        condition_negative_loss = zero
        if not self.scheme4_adaptive_lookahead_enabled:
            utility_loss = zero
            utility_benefit = zero.detach()
            utility_target = zero.detach()
        delta_token_metrics: dict[str, torch.Tensor] = {}
        future_metrics: dict[str, torch.Tensor] = {}
        condition_metrics: dict[str, torch.Tensor] = {}
        current_as_future_loss = zero.detach()
        oracle_future_loss = zero.detach()
        teacher_future_hidden = None
        needs_future_teacher = (
            self.scheme4_delta_token_loss_coeff != 0.0
            or self.scheme4_future_recon_loss_coeff != 0.0
            or self.scheme4_condition_negative_loss_coeff != 0.0
        )
        if self.scheme4_delta_enabled and needs_future_teacher:
            teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)
        if (
            self.scheme4_delta_enabled
            and (self.scheme4_delta_token_loss_coeff != 0.0 or self.scheme4_condition_negative_loss_coeff != 0.0)
            and teacher_future_hidden is not None
        ):
            teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
            target_delta_tokens, oracle_future_recon = self._scheme4_teacher_delta_targets(
                teacher_current_hidden,
                teacher_future_hidden,
                vision_mask,
            )
            delta_token_loss, delta_token_metrics = self._scheme4_delta_token_loss(
                pred_delta_tokens,
                target_delta_tokens,
            )
            condition_negative_loss, condition_metrics = self._scheme4_condition_negative_loss(
                pred_delta_tokens,
                target_delta_tokens,
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=language_features,
                state=state,
                action_intent=action_condition,
            )
            with torch.no_grad():
                oracle_future_loss, _ = self._scheme4_future_loss(
                    oracle_future_recon,
                    teacher_future_hidden,
                    vision_mask,
                )
        if (
            self.scheme4_delta_enabled
            and self.scheme4_future_recon_loss_coeff != 0.0
            and teacher_future_hidden is not None
        ):
            pred_future_tokens = self._scheme4_decode_future(spatial_tokens, pred_delta_tokens, vision_mask)
            future_recon_loss, future_metrics = self._scheme4_future_loss(
                pred_future_tokens,
                teacher_future_hidden,
                vision_mask,
            )
            with torch.no_grad():
                current_as_future_loss, _ = self._scheme4_future_loss(
                    spatial_tokens.detach(),
                    teacher_future_hidden,
                    vision_mask,
                )

        loss = (
            flow_loss
            + self.current_geo_loss_coeff * current_geo_loss
            + self.scheme4_delta_token_loss_coeff * delta_token_loss
            + self.scheme4_future_recon_loss_coeff * future_recon_loss
            + self.scheme4_condition_negative_loss_coeff * condition_negative_loss
            + self.scheme4_utility_loss_coeff * utility_loss
        )
        spatial_gate = self.scheme2_adapter.gate.float()
        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "future_recon_loss": future_recon_loss,
            "delta_token_loss": delta_token_loss,
            "scheme4_condition_negative_loss": condition_negative_loss,
            "spatial_gate_abs_mean": spatial_gate.abs().mean().detach(),
            "spatial_gate_max_abs": spatial_gate.abs().max().detach(),
            "delta_gate_abs_mean": self.scheme4_delta_adapter.gate_abs_mean().detach(),
            "delta_gate_max_abs": self.scheme4_delta_adapter.gate_max_abs().detach(),
            "delta_token_norm": pred_delta_tokens.float().norm(dim=-1).mean().detach(),
            "delta_residual_norm": delta_residual.float().norm(dim=-1).mean().detach(),
            "language_summary_norm": language_features.float().norm(dim=-1).mean().detach(),
            "action_intent_norm": action_intent.float().norm(dim=-1).mean().detach(),
            "action_condition_norm": action_condition.float().norm(dim=-1).mean().detach(),
            "language_token_residual_norm": torch.tensor(0.0, dtype=torch.float32, device=loss.device),
            "scheme4_delta_enabled": torch.tensor(
                float(self.scheme4_delta_enabled), dtype=torch.float32, device=loss.device
            ),
            "scheme4_utility_loss": utility_loss.detach(),
            "scheme4_utility_benefit_mean": utility_benefit.float().mean().detach(),
            "scheme4_utility_target_mean": utility_target.float().mean().detach(),
            "scheme4_router_probability_mean": lookahead_probability.float().mean().detach()
            if lookahead_probability is not None
            else torch.tensor(1.0, dtype=torch.float32, device=loss.device),
            "scheme4_router_selected_fraction": (lookahead_probability >= 0.5).float().mean().detach()
            if lookahead_probability is not None
            else torch.tensor(1.0, dtype=torch.float32, device=loss.device),
            "scheme4_training_intent_t1": torch.tensor(
                float(self.scheme4_training_intent_timestep == "t1_noise"),
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme4_spatial_norm": spatial_tokens.float().norm(dim=-1).mean().detach(),
            "scheme4_aux_abs_mean": aux_tokens.float().abs().mean().detach(),
            "scheme4_current_as_future_loss": current_as_future_loss.detach(),
            "scheme4_oracle_future_loss": oracle_future_loss.detach(),
            "scheme4_future_fallback_ratio": future_teacher_fallback.float().mean().detach()
            if future_teacher_fallback is not None
            else torch.tensor(0.0, dtype=torch.float32, device=loss.device),
            "scheme4_current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
            "scheme4_delta_token_loss_coeff": torch.tensor(
                self.scheme4_delta_token_loss_coeff,
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme4_future_recon_loss_coeff": torch.tensor(
                self.scheme4_future_recon_loss_coeff,
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme4_condition_negative_loss_coeff": torch.tensor(
                self.scheme4_condition_negative_loss_coeff,
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme4_condition_negative_margin": torch.tensor(
                self.scheme4_condition_negative_margin,
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme2_unified_spatial_loss_skipped": spatial_loss_skipped.detach(),
        }
        output.update(spatial_metrics)
        output.update(delta_token_metrics)
        output.update(future_metrics)
        output.update(condition_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        self._scheme4_latest_inference_diagnostics = None
        self._scheme4_inference_diagnostic_steps = [] if self.scheme4_inference_diagnostics_enabled else None
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        (
            images,
            _img_wo_aug,
            _teacher_images,
            _future_teacher_images,
            _future_teacher_fallback,
            img_padding_mask,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        _, _, prefix_hidden_states = self._prefix_hidden_from_embeddings(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        prefix_hidden = prefix_hidden_states[self.vla_layers_align].detach()
        vision_hidden = prefix_hidden[:, :img_len, :].detach()
        language_features = self._scheme4_language_summary(prefix_hidden, prefix_pad_masks, img_len).detach()
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        spatial_tokens, aux_tokens, aux_mask = self.scheme2_adapter(
            vision_hidden,
            vision_mask,
            select_aux=self.scheme2_injection_mode == "concat",
        )
        prefix_embs_aug, prefix_pad_masks_aug, prefix_att_masks_aug = self._inject_aux_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            aux_tokens,
            aux_mask,
        )
        _, past_key_values = self._prefix_hidden_from_embeddings(
            prefix_embs_aug,
            prefix_pad_masks_aug,
            prefix_att_masks_aug,
            use_cache=True,
            output_hidden_states=False,
        )

        if self.scheme4_inference_lookahead_mode == "none":
            # The no-lookahead intervention skips both the auxiliary intent
            # pass and future predictor while retaining the spatial prefix.
            action_intent = torch.zeros(bsize, self.action_out_proj.in_features, dtype=torch.float32, device=device)
            action_condition = action_intent
            delta_language_features = language_features
            delta_action_intent = action_intent
        else:
            initial_time = torch.ones(bsize, dtype=torch.float32, device=device)
            suffix_out = self._scheme4_delta_suffix_hidden_from_cache(
                state,
                prefix_pad_masks_aug,
                past_key_values,
                noise,
                initial_time,
            )
            action_intent = self._scheme4_action_intent_features(suffix_out)
            action_condition = apply_scheme4_training_action_condition(
                self.scheme4_training_action_condition,
                action_intent,
            )
            delta_language_features, delta_action_intent = self._scheme4_apply_inference_condition_control(
                language_features,
                action_condition,
            )
        lookahead_probability = None
        lookahead_scale = None
        if self.scheme4_inference_lookahead_mode == "none":
            # Explicit no-lookahead avoids running the learned future predictor.
            pred_delta_tokens = spatial_tokens.new_zeros(
                bsize,
                self.scheme4_tokenizer.config.num_delta_tokens,
                self.scheme4_tokenizer.config.delta_dim,
            )
            normal_pred_delta_tokens = pred_delta_tokens
            lookahead_probability = torch.zeros(bsize, dtype=torch.float32, device=device)
            lookahead_scale = lookahead_probability
        else:
            normal_pred_delta_tokens = self._scheme4_predict_delta(
                spatial_tokens,
                vision_mask=vision_mask,
                language_features=delta_language_features,
                state=state,
                action_intent=delta_action_intent,
            )
            pred_delta_tokens = self._scheme4_apply_inference_delta_control(normal_pred_delta_tokens)
            if self.scheme4_adaptive_lookahead_enabled:
                if self.scheme4_inference_lookahead_mode in {"soft", "hard"}:
                    lookahead_probability = self._scheme4_lookahead_probability(
                        spatial_tokens,
                        vision_mask,
                        delta_language_features,
                        state,
                        delta_action_intent,
                    )
                else:
                    lookahead_probability = torch.ones(bsize, dtype=torch.float32, device=device)
                lookahead_scale = resolve_lookahead_scale(
                    lookahead_probability,
                    self.scheme4_inference_lookahead_mode,
                )
        self._scheme4_update_inference_control_cache(
            language_features,
            action_condition,
            normal_pred_delta_tokens,
        )
        geo_cache = {
            "spatial_tokens": spatial_tokens.float(),
            "vision_mask": vision_mask,
            "language_features": delta_language_features.float(),
            "action_intent": delta_action_intent.float(),
            "pred_delta_tokens": pred_delta_tokens.float(),
        }
        if lookahead_scale is not None:
            geo_cache["lookahead_probability"] = lookahead_probability.float()
            geo_cache["fusion_scale"] = lookahead_scale.float() * self.scheme4_inference_fusion_scale

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks_aug, past_key_values, x_t, time.expand(bsize), geo_cache)
            x_t = x_t + dt * v_t
            time += dt
        if self._scheme4_inference_diagnostic_steps is not None:
            self._scheme4_latest_inference_diagnostics = finalize_request_diagnostics(
                self._scheme4_inference_diagnostic_steps,
                pred_delta_tokens=pred_delta_tokens,
                action_intent=delta_action_intent,
                fusion_scale=self.scheme4_inference_fusion_scale,
            )
            if lookahead_scale is not None:
                self._scheme4_latest_inference_diagnostics["lookahead_probability"] = (
                    lookahead_probability.float().detach()
                )
                self._scheme4_latest_inference_diagnostics["lookahead_scale"] = lookahead_scale.float().detach()
                self._scheme4_latest_inference_diagnostics["fusion_scale"] = geo_cache["fusion_scale"].float().detach()
            self._scheme4_inference_diagnostic_steps = None
        return x_t

    def take_scheme4_inference_diagnostics(self) -> dict[str, torch.Tensor] | None:
        """Return and clear the latest opt-in detached inference diagnostics."""

        diagnostics = self._scheme4_latest_inference_diagnostics
        self._scheme4_latest_inference_diagnostics = None
        return diagnostics

    def _scheme4_delta_suffix_hidden_from_cache(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        expected = (batch_size, self.config.action_horizon, self.action_out_proj.in_features)
        if suffix_out.shape != expected:
            raise RuntimeError(f"suffix_out shape {tuple(suffix_out.shape)} does not match {expected}")
        return suffix_out

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, geo_cache=None):
        if geo_cache is None or "pred_delta_tokens" not in geo_cache:
            raise ValueError("scheme4_lac_double_align denoise_step requires geo_cache with pred_delta_tokens")
        suffix_out = self._scheme4_delta_suffix_hidden_from_cache(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        )
        fusion_scale = geo_cache.get("fusion_scale", self.scheme4_inference_fusion_scale)
        v_t, _pred_delta_tokens, delta_residual = self._apply_scheme4_delta_fusion(
            suffix_out,
            geo_cache["spatial_tokens"],
            vision_mask=geo_cache["vision_mask"],
            language_features=geo_cache["language_features"],
            state=state,
            action_intent=geo_cache["action_intent"],
            pred_delta_tokens=geo_cache["pred_delta_tokens"],
            fusion_scale=fusion_scale,
        )
        if self._scheme4_inference_diagnostic_steps is not None:
            base_v_t = self._apply_checkpoint(lambda hidden: self.action_out_proj(hidden), suffix_out)
            self._scheme4_inference_diagnostic_steps.append(
                flow_step_diagnostics(suffix_out, delta_residual, v_t, base_v_t)
            )
        return v_t


PROPOSAL_VARIANTS = {
    "scheme1": Scheme1SharedMLPPI0Pytorch,
    "scheme1_rq": Scheme1ResidualQueryPI0Pytorch,
    "scheme2": Scheme2VisualConcatPI0Pytorch,
    "scheme2_g": Scheme2GatedDualHeadPI0Pytorch,
    "scheme2_unified": Scheme2UnifiedSpatialPI0Pytorch,
    "scheme2_unified_aux": Scheme2UnifiedSpatialAuxOnlyPI0Pytorch,
    "direct_vggt_injection": DirectVGGTInjectionPI0Pytorch,
    "scheme3_d3": Scheme3D3FusionPI0Pytorch,
    "scheme3_delta": Scheme3CorrectedDeltaPI0Pytorch,
    "scheme3_delta_intent": Scheme3CorrectedDeltaActionIntentPI0Pytorch,
    "scheme4_lac_double_align": Scheme4LACDoubleAlignPI0Pytorch,
}

LAZY_PROPOSAL_VARIANTS = (
    "scheme4_grounded_lac_double_align",
    "scheme4_frame_token_lac_double_align",
)


def get_proposal_model_class(name: str):
    if name == "scheme4_grounded_lac_double_align":
        from openpi.models_pytorch.scheme4_grounded_lac_pytorch import Scheme4GroundedLACDoubleAlignPI0Pytorch

        return Scheme4GroundedLACDoubleAlignPI0Pytorch
    if name == "scheme4_frame_token_lac_double_align":
        from openpi.models_pytorch.scheme4_frame_token_lac_pytorch import (  # noqa: I001, PLC0415
            Scheme4FrameTokenLACDoubleAlignPI0Pytorch,
        )

        return Scheme4FrameTokenLACDoubleAlignPI0Pytorch
    try:
        return PROPOSAL_VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(sorted((*PROPOSAL_VARIANTS, *LAZY_PROPOSAL_VARIANTS)))
        raise ValueError(f"Unknown proposal variant {name!r}. Available: {available}") from exc
