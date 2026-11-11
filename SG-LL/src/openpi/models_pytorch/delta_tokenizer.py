"""Decodable geometry delta tokenizer for scheme3.

This module is intentionally standalone: it does not import pi0 policy classes
and does not change the action path.  The expected inputs are clean teacher
geometry tokens, for example VGGT tokens already aligned to pi visual tokens.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class DeltaTokenizerConfig:
    """Configuration for a decodable geometry transition tokenizer."""

    teacher_dim: int = 2048
    hidden_dim: int = 512
    delta_dim: int = 256
    num_delta_tokens: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    max_tokens: int = 1024
    max_horizon: int = 256
    num_camera_embeddings: int = 0
    relation_max_tokens: int = 128
    zero_init_decoder: bool = True

    def __post_init__(self) -> None:
        positive_fields = {
            "teacher_dim": self.teacher_dim,
            "hidden_dim": self.hidden_dim,
            "delta_dim": self.delta_dim,
            "num_delta_tokens": self.num_delta_tokens,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "num_heads": self.num_heads,
            "max_tokens": self.max_tokens,
            "max_horizon": self.max_horizon,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")
        if self.num_camera_embeddings < 0:
            raise ValueError(f"num_camera_embeddings must be >= 0, got {self.num_camera_embeddings}")
        if self.relation_max_tokens <= 0:
            raise ValueError(f"relation_max_tokens must be positive, got {self.relation_max_tokens}")


@dataclasses.dataclass(frozen=True)
class DeltaTokenizerOutput:
    """Forward output.

    Shapes:
    - delta_tokens: [B, M_delta, D_delta]
    - future_recon: [B, N_teacher, D_teacher]
    """

    delta_tokens: torch.Tensor
    future_recon: torch.Tensor


@dataclasses.dataclass(frozen=True)
class DeltaTokenizerLossOutput:
    """Loss output for standalone tokenizer training."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    model_output: DeltaTokenizerOutput


def _assert_rank3(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B, N, D], got {tuple(value.shape)}")


def _check_teacher_pair(current_tokens: torch.Tensor, future_tokens: torch.Tensor) -> tuple[int, int, int]:
    _assert_rank3("current_tokens", current_tokens)
    _assert_rank3("future_tokens", future_tokens)
    if current_tokens.shape != future_tokens.shape:
        raise ValueError(
            "current_tokens and future_tokens must have identical shape, "
            f"got {tuple(current_tokens.shape)} and {tuple(future_tokens.shape)}"
        )
    return int(current_tokens.shape[0]), int(current_tokens.shape[1]), int(current_tokens.shape[2])


def _check_token_mask(
    token_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if token_mask is None:
        return None
    if token_mask.shape != (batch_size, num_tokens):
        raise ValueError(f"token_mask must have shape [{batch_size}, {num_tokens}], got {tuple(token_mask.shape)}")
    if token_mask.device != device:
        raise ValueError(f"token_mask device {token_mask.device} does not match token device {device}")
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
    if horizon_ids.ndim != 1 or horizon_ids.shape[0] != batch_size:
        raise ValueError(f"horizon_ids must have shape [{batch_size}], got {tuple(horizon_ids.shape)}")
    if horizon_ids.device != device:
        raise ValueError(f"horizon_ids device {horizon_ids.device} does not match token device {device}")
    if not torch.is_floating_point(horizon_ids) and not torch.is_complex(horizon_ids):
        ids = horizon_ids.long()
    else:
        raise ValueError(f"horizon_ids must be an integer tensor, got dtype={horizon_ids.dtype}")
    if ids.numel() > 0 and (int(ids.min()) < 0 or int(ids.max()) >= max_horizon):
        raise ValueError(f"horizon_ids must be in [0, {max_horizon}), got min={int(ids.min())}, max={int(ids.max())}")
    return ids


def _check_camera_ids(
    camera_ids: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    max_cameras: int,
    device: torch.device,
) -> torch.Tensor | None:
    if camera_ids is None:
        return None
    if max_cameras <= 0:
        raise ValueError("camera_ids were provided but num_camera_embeddings is 0")
    if camera_ids.shape != (batch_size, num_tokens):
        raise ValueError(f"camera_ids must have shape [{batch_size}, {num_tokens}], got {tuple(camera_ids.shape)}")
    if camera_ids.device != device:
        raise ValueError(f"camera_ids device {camera_ids.device} does not match token device {device}")
    if torch.is_floating_point(camera_ids) or torch.is_complex(camera_ids):
        raise ValueError(f"camera_ids must be an integer tensor, got dtype={camera_ids.dtype}")
    ids = camera_ids.long()
    if ids.numel() > 0 and (int(ids.min()) < 0 or int(ids.max()) >= max_cameras):
        raise ValueError(f"camera_ids must be in [0, {max_cameras}), got min={int(ids.min())}, max={int(ids.max())}")
    return ids


def _expand_to_tokens(value: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return value.expand(value.shape[0], num_tokens, value.shape[-1])


def _masked_token_mean(loss: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return loss.mean()
    mask = token_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class _CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block with explicit mask handling."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _assert_rank3("query", query)
        _assert_rank3("memory", memory)
        if query.shape[0] != memory.shape[0] or query.shape[-1] != memory.shape[-1]:
            raise ValueError(
                "query and memory must share batch and hidden dimensions, "
                f"got {tuple(query.shape)} and {tuple(memory.shape)}"
            )
        if memory_mask is not None and memory_mask.shape != memory.shape[:2]:
            raise ValueError(f"memory_mask must have shape {tuple(memory.shape[:2])}, got {tuple(memory_mask.shape)}")

        key_padding_mask = None if memory_mask is None else ~memory_mask.bool()
        query = query.float()
        memory = memory.float()
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.attn_dropout(attn_out)
        return query + self.ffn(query)


class _SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block for learned delta queries."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _assert_rank3("tokens", tokens)
        tokens = tokens.float()
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.attn_dropout(attn_out)
        return tokens + self.ffn(tokens)


class _DeltaEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = _SelfAttentionBlock(dim, num_heads, dropout)
        self.cross_attn = _CrossAttentionBlock(dim, num_heads, dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query_tokens = self.self_attn(query_tokens)
        return self.cross_attn(query_tokens, memory, memory_mask)


class DeltaEncoder(nn.Module):
    """Encode clean current/future teacher tokens into compact delta tokens.

    Input shapes:
    - current_tokens: [B, N, D_teacher]
    - future_tokens: [B, N, D_teacher]
    - token_mask: optional [B, N]
    - horizon_ids: optional [B]
    - camera_ids: optional [B, N]

    Output shape:
    - delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(self, config: DeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.teacher_dim, config.hidden_dim)
        self.current_type = nn.Parameter(torch.randn(1, 1, config.hidden_dim) / math.sqrt(config.hidden_dim))
        self.future_type = nn.Parameter(torch.randn(1, 1, config.hidden_dim) / math.sqrt(config.hidden_dim))
        self.position_embed = nn.Parameter(
            torch.randn(config.max_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.delta_queries = nn.Parameter(
            torch.randn(config.num_delta_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.camera_embed = (
            nn.Embedding(config.num_camera_embeddings, config.hidden_dim) if config.num_camera_embeddings > 0 else None
        )
        self.layers = nn.ModuleList(
            [
                _DeltaEncoderLayer(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.encoder_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.delta_dim),
        )

    def _horizon_features(
        self,
        horizon_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = _check_horizon_ids(horizon_ids, batch_size=batch_size, max_horizon=self.config.max_horizon, device=device)
        if ids is None:
            return torch.zeros(batch_size, 1, self.config.hidden_dim, dtype=dtype, device=device)
        return self.horizon_embed(ids).to(dtype=dtype).unsqueeze(1)

    def _camera_features(
        self,
        camera_ids: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = _check_camera_ids(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            max_cameras=self.config.num_camera_embeddings,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, num_tokens, self.config.hidden_dim, dtype=dtype, device=device)
        if self.camera_embed is None:
            raise RuntimeError("camera_embed is unexpectedly None")
        return self.camera_embed(ids).to(dtype=dtype)

    def _build_memory(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        horizon_ids: torch.Tensor | None,
        camera_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, num_tokens, teacher_dim = _check_teacher_pair(current_tokens, future_tokens)
        if teacher_dim != self.config.teacher_dim:
            raise ValueError(f"teacher dim {teacher_dim} does not match config.teacher_dim={self.config.teacher_dim}")
        if num_tokens > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.config.max_tokens}")

        device = current_tokens.device
        token_mask = _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=device)
        horizon = self._horizon_features(
            horizon_ids,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        horizon = _expand_to_tokens(horizon, num_tokens)
        position = self.position_embed[:num_tokens].to(dtype=torch.float32, device=device).unsqueeze(0)
        position = position.expand(batch_size, num_tokens, self.config.hidden_dim)
        camera = self._camera_features(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device,
            dtype=torch.float32,
        )
        current_type = self.current_type.to(dtype=torch.float32, device=device).expand(
            batch_size, num_tokens, self.config.hidden_dim
        )
        future_type = self.future_type.to(dtype=torch.float32, device=device).expand(
            batch_size, num_tokens, self.config.hidden_dim
        )

        current = self.input_proj(current_tokens.float()) + current_type + position + horizon + camera
        future = self.input_proj(future_tokens.float()) + future_type + position + horizon + camera
        memory = torch.cat([current, future], dim=1)
        memory_mask = None if token_mask is None else torch.cat([token_mask, token_mask], dim=1)
        return memory, memory_mask

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_mask = self._build_memory(current_tokens, future_tokens, token_mask, horizon_ids, camera_ids)
        batch_size = current_tokens.shape[0]
        horizon = self._horizon_features(
            horizon_ids,
            batch_size=batch_size,
            device=current_tokens.device,
            dtype=torch.float32,
        )
        horizon = _expand_to_tokens(horizon, self.config.num_delta_tokens)
        queries = self.delta_queries.to(dtype=torch.float32, device=current_tokens.device).unsqueeze(0)
        queries = queries.expand(batch_size, self.config.num_delta_tokens, self.config.hidden_dim) + horizon
        for layer in self.layers:
            queries = layer(queries, memory, memory_mask)
        return self.out(queries)


class DeltaDecoder(nn.Module):
    """Decode future teacher tokens from current teacher tokens and delta tokens.

    Input shapes:
    - current_tokens: [B, N, D_teacher]
    - delta_tokens: [B, M_delta, D_delta]

    Output shape:
    - future_recon: [B, N, D_teacher]
    """

    def __init__(self, config: DeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.current_proj = nn.Linear(config.teacher_dim, config.hidden_dim)
        self.delta_proj = nn.Linear(config.delta_dim, config.hidden_dim)
        self.position_embed = nn.Parameter(
            torch.randn(config.max_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.camera_embed = (
            nn.Embedding(config.num_camera_embeddings, config.hidden_dim) if config.num_camera_embeddings > 0 else None
        )
        self.layers = nn.ModuleList(
            [
                _CrossAttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.decoder_layers)
            ]
        )
        self.residual_out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.teacher_dim),
        )
        if config.zero_init_decoder:
            nn.init.zeros_(self.residual_out[-1].weight)
            nn.init.zeros_(self.residual_out[-1].bias)

    def _horizon_features(
        self,
        horizon_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = _check_horizon_ids(horizon_ids, batch_size=batch_size, max_horizon=self.config.max_horizon, device=device)
        if ids is None:
            return torch.zeros(batch_size, 1, self.config.hidden_dim, dtype=dtype, device=device)
        return self.horizon_embed(ids).to(dtype=dtype).unsqueeze(1)

    def _camera_features(
        self,
        camera_ids: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = _check_camera_ids(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            max_cameras=self.config.num_camera_embeddings,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, num_tokens, self.config.hidden_dim, dtype=dtype, device=device)
        if self.camera_embed is None:
            raise RuntimeError("camera_embed is unexpectedly None")
        return self.camera_embed(ids).to(dtype=dtype)

    def forward(
        self,
        current_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _assert_rank3("current_tokens", current_tokens)
        _assert_rank3("delta_tokens", delta_tokens)
        batch_size, num_tokens, teacher_dim = current_tokens.shape
        if teacher_dim != self.config.teacher_dim:
            raise ValueError(f"teacher dim {teacher_dim} does not match config.teacher_dim={self.config.teacher_dim}")
        if delta_tokens.shape[0] != batch_size:
            raise ValueError(f"delta_tokens batch {delta_tokens.shape[0]} does not match current batch {batch_size}")
        if delta_tokens.shape[-1] != self.config.delta_dim:
            raise ValueError(
                f"delta dim {delta_tokens.shape[-1]} does not match config.delta_dim={self.config.delta_dim}"
            )
        if num_tokens > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.config.max_tokens}")

        device = current_tokens.device
        _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=device)
        horizon = self._horizon_features(horizon_ids, batch_size=batch_size, device=device, dtype=torch.float32)
        query_horizon = _expand_to_tokens(horizon, num_tokens)
        delta_horizon = _expand_to_tokens(horizon, delta_tokens.shape[1])
        position = self.position_embed[:num_tokens].to(dtype=torch.float32, device=device).unsqueeze(0)
        position = position.expand(batch_size, num_tokens, self.config.hidden_dim)
        camera = self._camera_features(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device,
            dtype=torch.float32,
        )

        query = self.current_proj(current_tokens.float()) + position + query_horizon + camera
        memory = self.delta_proj(delta_tokens.float()) + delta_horizon
        for layer in self.layers:
            query = layer(query, memory)
        residual = self.residual_out(query.float())
        return current_tokens.float() + residual


class DeltaTokenizer(nn.Module):
    """Encoder/decoder wrapper for decodable geometry delta tokens."""

    def __init__(self, config: DeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DeltaEncoder(config)
        self.decoder = DeltaDecoder(config)

    def encode(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            current_tokens,
            future_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def decode(
        self,
        current_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            current_tokens,
            delta_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> DeltaTokenizerOutput:
        delta_tokens = self.encode(
            current_tokens,
            future_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        future_recon = self.decode(
            current_tokens,
            delta_tokens,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        return DeltaTokenizerOutput(delta_tokens=delta_tokens, future_recon=future_recon)

    def config_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self.config)


def geometry_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    relation_weight: float = 0.05,
    relation_max_tokens: int = 128,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Geometry distance used for tokenizer reconstruction.

    Shapes:
    - pred: [B, N, D]
    - target: [B, N, D]
    - token_mask: optional [B, N]
    """

    batch_size, num_tokens, _ = _check_teacher_pair(pred, target)
    token_mask = _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=pred.device)
    value_per_token = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    value_loss = _masked_token_mean(value_per_token, token_mask)

    pred_norm = F.normalize(pred.float(), dim=-1)
    target_norm = F.normalize(target.float(), dim=-1)
    cosine_per_token = 1.0 - (pred_norm * target_norm).sum(dim=-1)
    cosine_loss = _masked_token_mean(cosine_per_token, token_mask)

    relation_loss = _relation_loss(pred_norm, target_norm, token_mask, max_tokens=relation_max_tokens)
    loss = (
        float(smooth_l1_weight) * value_loss
        + float(cosine_weight) * cosine_loss
        + float(relation_weight) * relation_loss
    )
    return loss, {
        "geometry_value_loss": value_loss.detach(),
        "geometry_cosine_loss": cosine_loss.detach(),
        "geometry_relation_loss": relation_loss.detach(),
    }


def _relation_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    token_mask: torch.Tensor | None,
    *,
    max_tokens: int,
) -> torch.Tensor:
    if pred_norm.shape[1] > max_tokens:
        index = torch.linspace(0, pred_norm.shape[1] - 1, steps=max_tokens, device=pred_norm.device).round().long()
        pred_norm = pred_norm.index_select(1, index)
        target_norm = target_norm.index_select(1, index)
        if token_mask is not None:
            token_mask = token_mask.index_select(1, index)
    pred_sim = torch.matmul(pred_norm, pred_norm.transpose(1, 2))
    target_sim = torch.matmul(target_norm, target_norm.transpose(1, 2))
    loss = (pred_sim - target_sim).abs()
    if token_mask is None:
        return loss.mean()
    pair_mask = token_mask.bool().unsqueeze(2) & token_mask.bool().unsqueeze(1)
    pair_mask_float = pair_mask.float()
    return (loss * pair_mask_float).sum() / pair_mask_float.sum().clamp_min(1.0)


def delta_tokenizer_loss(
    tokenizer: DeltaTokenizer,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    horizon_ids: torch.Tensor | None = None,
    camera_ids: torch.Tensor | None = None,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    relation_weight: float = 0.05,
    identity_weight: float = 0.1,
) -> DeltaTokenizerLossOutput:
    """Compute standalone tokenizer reconstruction and identity losses."""

    output = tokenizer(
        current_tokens,
        future_tokens,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        camera_ids=camera_ids,
    )
    rec_loss, rec_metrics = geometry_distance(
        output.future_recon,
        future_tokens.detach(),
        token_mask,
        smooth_l1_weight=smooth_l1_weight,
        cosine_weight=cosine_weight,
        relation_weight=relation_weight,
        relation_max_tokens=tokenizer.config.relation_max_tokens,
    )

    zero = rec_loss * 0.0
    identity_recon_loss = zero
    identity_delta_loss = zero
    if float(identity_weight) > 0.0:
        zero_delta = torch.zeros_like(output.delta_tokens)
        identity_recon = tokenizer.decode(
            current_tokens,
            zero_delta,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        identity_recon_loss, _ = geometry_distance(
            identity_recon,
            current_tokens.detach(),
            token_mask,
            smooth_l1_weight=smooth_l1_weight,
            cosine_weight=cosine_weight,
            relation_weight=relation_weight,
            relation_max_tokens=tokenizer.config.relation_max_tokens,
        )
        same_delta = tokenizer.encode(
            current_tokens,
            current_tokens.detach(),
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        identity_delta_loss = same_delta.float().abs().mean()

    loss = rec_loss + float(identity_weight) * (identity_recon_loss + identity_delta_loss)
    metrics = {
        "delta_tokenizer_loss": loss.detach(),
        "delta_rec_loss": rec_loss.detach(),
        "delta_identity_recon_loss": identity_recon_loss.detach(),
        "delta_identity_token_l1": identity_delta_loss.detach(),
        **rec_metrics,
    }
    return DeltaTokenizerLossOutput(loss=loss, metrics=metrics, model_output=output)
