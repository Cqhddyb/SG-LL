"""Language/action conditioned geometry delta tokenizer for scheme4.

This module is teacher-side representation infrastructure.  It does not import
pi0 policy classes and does not change the action path by itself.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerLossOutput
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerOutput
from openpi.models_pytorch.delta_tokenizer import _assert_rank3
from openpi.models_pytorch.delta_tokenizer import _check_camera_ids
from openpi.models_pytorch.delta_tokenizer import _check_horizon_ids
from openpi.models_pytorch.delta_tokenizer import _check_teacher_pair
from openpi.models_pytorch.delta_tokenizer import _check_token_mask
from openpi.models_pytorch.delta_tokenizer import _CrossAttentionBlock
from openpi.models_pytorch.delta_tokenizer import _DeltaEncoderLayer
from openpi.models_pytorch.delta_tokenizer import _expand_to_tokens
from openpi.models_pytorch.delta_tokenizer import geometry_distance


@dataclasses.dataclass(frozen=True)
class ConditionalDeltaTokenizerConfig:
    """Configuration for a conditional decodable geometry transition tokenizer."""

    teacher_dim: int = 2048
    language_dim: int = 2048
    action_horizon: int = 10
    action_dim: int = 32
    state_dim: int = 32
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
            "language_dim": self.language_dim,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
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


def _check_rank2_condition(
    name: str,
    value: torch.Tensor,
    *,
    batch_size: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if value.shape != (batch_size, dim):
        raise ValueError(f"{name} must have shape [{batch_size}, {dim}], got {tuple(value.shape)}")
    if value.device != device:
        raise ValueError(f"{name} device {value.device} does not match token device {device}")
    return value.float()


def _check_actions(
    actions: torch.Tensor,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    expected = (batch_size, action_horizon, action_dim)
    if actions.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")
    if actions.device != device:
        raise ValueError(f"actions device {actions.device} does not match token device {device}")
    return actions.float()


class _ConditionalContextProjector(nn.Module):
    """Project language, action, state, and horizon into one hidden condition."""

    def __init__(self, config: ConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.language_proj = nn.Sequential(
            nn.LayerNorm(config.language_dim), nn.Linear(config.language_dim, config.hidden_dim)
        )
        action_flat_dim = config.action_horizon * config.action_dim
        self.action_proj = nn.Sequential(nn.LayerNorm(action_flat_dim), nn.Linear(action_flat_dim, config.hidden_dim))
        self.state_proj = nn.Sequential(nn.LayerNorm(config.state_dim), nn.Linear(config.state_dim, config.hidden_dim))
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        horizon_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        language_features = _check_rank2_condition(
            "language_features",
            language_features,
            batch_size=batch_size,
            dim=self.config.language_dim,
            device=device,
        )
        actions = _check_actions(
            actions,
            batch_size=batch_size,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            device=device,
        )
        state = _check_rank2_condition(
            "state",
            state,
            batch_size=batch_size,
            dim=self.config.state_dim,
            device=device,
        )
        ids = _check_horizon_ids(
            horizon_ids,
            batch_size=batch_size,
            max_horizon=self.config.max_horizon,
            device=device,
        )
        if ids is None:
            horizon = torch.zeros(batch_size, self.config.hidden_dim, dtype=torch.float32, device=device)
        else:
            horizon = self.horizon_embed(ids).float()

        action_flat = actions.reshape(batch_size, self.config.action_horizon * self.config.action_dim)
        cond = self.language_proj(language_features) + self.action_proj(action_flat) + self.state_proj(state) + horizon
        return self.out(cond.float())


class ConditionalDeltaEncoder(nn.Module):
    """Encode current/future teacher tokens into conditional compact delta tokens.

    Shape contract:
    - current_tokens: [B, N, D_teacher]
    - future_tokens: [B, N, D_teacher]
    - language_features: [B, D_language]
    - actions: [B, H_action, A_dim]
    - state: [B, D_state]
    - token_mask: optional [B, N]
    - horizon_ids: optional [B]
    - camera_ids: optional [B, N]
    - output delta_tokens: [B, M_delta, D_delta]
    """

    def __init__(self, config: ConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = _ConditionalContextProjector(config)
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
        self.cond_to_queries = nn.Linear(config.hidden_dim, config.num_delta_tokens * config.hidden_dim)
        self.cond_to_memory = nn.Linear(config.hidden_dim, config.hidden_dim)
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
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, num_tokens, teacher_dim = _check_teacher_pair(current_tokens, future_tokens)
        if teacher_dim != self.config.teacher_dim:
            raise ValueError(f"teacher dim {teacher_dim} does not match config.teacher_dim={self.config.teacher_dim}")
        if num_tokens > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.config.max_tokens}")

        device = current_tokens.device
        token_mask = _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=device)
        horizon = self._horizon_features(horizon_ids, batch_size=batch_size, device=device, dtype=torch.float32)
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
        cond_memory = self.cond_to_memory(cond).view(batch_size, 1, self.config.hidden_dim)
        cond_memory = cond_memory.expand(batch_size, num_tokens, self.config.hidden_dim)

        current = self.input_proj(current_tokens.float()) + current_type + position + horizon + camera + cond_memory
        future = self.input_proj(future_tokens.float()) + future_type + position + horizon + camera + cond_memory
        memory = torch.cat([current, future], dim=1)
        memory_mask = None if token_mask is None else torch.cat([token_mask, token_mask], dim=1)
        return memory, memory_mask

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _num_tokens, _teacher_dim = _check_teacher_pair(current_tokens, future_tokens)
        cond = self.condition(
            batch_size=batch_size,
            device=current_tokens.device,
            language_features=language_features,
            actions=actions,
            state=state,
            horizon_ids=horizon_ids,
        )
        memory, memory_mask = self._build_memory(
            current_tokens,
            future_tokens,
            token_mask,
            horizon_ids,
            camera_ids,
            cond,
        )
        horizon = self._horizon_features(
            horizon_ids,
            batch_size=batch_size,
            device=current_tokens.device,
            dtype=torch.float32,
        )
        horizon = _expand_to_tokens(horizon, self.config.num_delta_tokens)
        base_queries = self.delta_queries.to(dtype=torch.float32, device=current_tokens.device).unsqueeze(0)
        base_queries = base_queries.expand(batch_size, self.config.num_delta_tokens, self.config.hidden_dim)
        cond_queries = self.cond_to_queries(cond).view(batch_size, self.config.num_delta_tokens, self.config.hidden_dim)
        queries = base_queries + horizon + cond_queries
        for layer in self.layers:
            queries = layer(queries, memory, memory_mask)
        return self.out(queries)


class ConditionalDeltaDecoder(nn.Module):
    """Decode future teacher tokens from current tokens, delta tokens, and condition.

    Shape contract:
    - current_tokens: [B, N, D_teacher]
    - delta_tokens: [B, M_delta, D_delta]
    - language_features: [B, D_language]
    - actions: [B, H_action, A_dim]
    - state: [B, D_state]
    - output future_recon: [B, N, D_teacher]
    """

    def __init__(self, config: ConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = _ConditionalContextProjector(config)
        self.current_proj = nn.Linear(config.teacher_dim, config.hidden_dim)
        self.delta_proj = nn.Linear(config.delta_dim, config.hidden_dim)
        self.position_embed = nn.Parameter(
            torch.randn(config.max_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.camera_embed = (
            nn.Embedding(config.num_camera_embeddings, config.hidden_dim) if config.num_camera_embeddings > 0 else None
        )
        self.cond_to_query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.cond_to_delta_memory = nn.Linear(config.hidden_dim, config.hidden_dim)
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
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
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
        cond = self.condition(
            batch_size=batch_size,
            device=device,
            language_features=language_features,
            actions=actions,
            state=state,
            horizon_ids=horizon_ids,
        )
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
        cond_query = self.cond_to_query(cond).view(batch_size, 1, self.config.hidden_dim)
        cond_query = cond_query.expand(batch_size, num_tokens, self.config.hidden_dim)
        cond_delta = self.cond_to_delta_memory(cond).view(batch_size, 1, self.config.hidden_dim)
        cond_delta = cond_delta.expand(batch_size, delta_tokens.shape[1], self.config.hidden_dim)

        query = self.current_proj(current_tokens.float()) + position + query_horizon + camera + cond_query
        memory = self.delta_proj(delta_tokens.float()) + delta_horizon + cond_delta
        for layer in self.layers:
            query = layer(query, memory)
        residual = self.residual_out(query.float())
        return current_tokens.float() + residual


class ConditionalDeltaTokenizer(nn.Module):
    """Encoder/decoder wrapper for conditional decodable geometry delta tokens."""

    def __init__(self, config: ConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConditionalDeltaEncoder(config)
        self.decoder = ConditionalDeltaDecoder(config)

    def encode(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            current_tokens,
            future_tokens,
            language_features=language_features,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def decode(
        self,
        current_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            current_tokens,
            delta_tokens,
            language_features=language_features,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> DeltaTokenizerOutput:
        delta_tokens = self.encode(
            current_tokens,
            future_tokens,
            language_features=language_features,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        future_recon = self.decode(
            current_tokens,
            delta_tokens,
            language_features=language_features,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        return DeltaTokenizerOutput(delta_tokens=delta_tokens, future_recon=future_recon)

    def config_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self.config)


def conditional_delta_tokenizer_loss(
    tokenizer: ConditionalDeltaTokenizer,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    language_features: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    horizon_ids: torch.Tensor | None = None,
    camera_ids: torch.Tensor | None = None,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    relation_weight: float = 0.05,
    identity_weight: float = 0.0,
) -> DeltaTokenizerLossOutput:
    """Train conditional tokenizer with future reconstruction and optional identity loss."""

    output = tokenizer(
        current_tokens,
        future_tokens,
        language_features=language_features,
        actions=actions,
        state=state,
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        camera_ids=camera_ids,
    )
    rec_loss, rec_metrics = geometry_distance(
        output.future_recon,
        future_tokens,
        token_mask,
        smooth_l1_weight=smooth_l1_weight,
        cosine_weight=cosine_weight,
        relation_weight=relation_weight,
        relation_max_tokens=tokenizer.config.relation_max_tokens,
    )
    loss = rec_loss
    metrics = {
        "conditional_delta_tokenizer_loss": rec_loss.detach(),
        "conditional_delta_rec_loss": rec_loss.detach(),
        "conditional_delta_abs_mean": output.delta_tokens.float().abs().mean().detach(),
        **{f"conditional_delta_{key}": value.detach() for key, value in rec_metrics.items()},
    }
    if identity_weight != 0.0:
        zero_delta = torch.zeros_like(output.delta_tokens)
        identity_recon = tokenizer.decode(
            current_tokens,
            zero_delta,
            language_features=language_features,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        identity_loss, identity_metrics = geometry_distance(
            identity_recon,
            current_tokens,
            token_mask,
            smooth_l1_weight=smooth_l1_weight,
            cosine_weight=cosine_weight,
            relation_weight=relation_weight,
            relation_max_tokens=tokenizer.config.relation_max_tokens,
        )
        loss = loss + float(identity_weight) * identity_loss
        metrics["conditional_delta_identity_loss"] = identity_loss.detach()
        metrics.update({f"conditional_delta_identity_{key}": value.detach() for key, value in identity_metrics.items()})
    metrics["conditional_delta_tokenizer_loss"] = loss.detach()
    return DeltaTokenizerLossOutput(loss=loss, metrics=metrics, model_output=output)
