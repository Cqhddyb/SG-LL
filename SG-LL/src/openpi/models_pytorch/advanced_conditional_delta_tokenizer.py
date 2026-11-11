"""Advanced language/action conditioned geometry delta tokenizer.

This is Route B infrastructure for Scheme4 diagnostics.  It keeps the
teacher-side tokenizer standalone and supports token-level language/action
cross-attention, FiLM/AdaLN conditioning, condition dropout, and optional
negative-condition loss.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

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
from openpi.models_pytorch.scheme4_condition_diagnostics import margin_negative_condition_loss
from openpi.models_pytorch.scheme4_condition_diagnostics import roll_batch_condition


@dataclasses.dataclass(frozen=True)
class AdvancedConditionalDeltaTokenizerConfig:
    """Configuration for Route B conditional tokenizer v2."""

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
    condition_dropout_prob: float = 0.0
    max_tokens: int = 1024
    max_language_tokens: int = 128
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
            "max_language_tokens": self.max_language_tokens,
            "max_horizon": self.max_horizon,
            "relation_max_tokens": self.relation_max_tokens,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}")
        if self.num_camera_embeddings < 0:
            raise ValueError(f"num_camera_embeddings must be >= 0, got {self.num_camera_embeddings}")
        if not 0.0 <= float(self.condition_dropout_prob) < 1.0:
            raise ValueError(f"condition_dropout_prob must be in [0, 1), got {self.condition_dropout_prob}")


def _check_language_tokens(
    language_token_features: torch.Tensor,
    language_token_mask: torch.Tensor | None,
    *,
    batch_size: int,
    language_dim: int,
    max_language_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if language_token_features.ndim != 3:
        raise ValueError(
            "language_token_features must have shape [B, L_language, D_language], "
            f"got {tuple(language_token_features.shape)}"
        )
    if language_token_features.shape[0] != batch_size or language_token_features.shape[-1] != language_dim:
        raise ValueError(
            "language_token_features must have shape "
            f"[{batch_size}, L_language, {language_dim}], got {tuple(language_token_features.shape)}"
        )
    if language_token_features.shape[1] > max_language_tokens:
        raise ValueError(
            f"language token count {language_token_features.shape[1]} exceeds max_language_tokens={max_language_tokens}"
        )
    if language_token_features.device != device:
        raise ValueError(
            f"language_token_features device {language_token_features.device} does not match token device {device}"
        )
    if language_token_mask is None:
        mask = torch.ones(
            batch_size,
            language_token_features.shape[1],
            dtype=torch.bool,
            device=device,
        )
    else:
        if language_token_mask.shape != language_token_features.shape[:2]:
            raise ValueError(
                "language_token_mask must have shape "
                f"{tuple(language_token_features.shape[:2])}, got {tuple(language_token_mask.shape)}"
            )
        if language_token_mask.device != device:
            raise ValueError(
                f"language_token_mask device {language_token_mask.device} does not match token device {device}"
            )
        mask = language_token_mask.bool()
    return language_token_features.float(), mask


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


def _check_state(
    state: torch.Tensor,
    *,
    batch_size: int,
    state_dim: int,
    device: torch.device,
) -> torch.Tensor:
    expected = (batch_size, state_dim)
    if state.shape != expected:
        raise ValueError(f"state must have shape {expected}, got {tuple(state.shape)}")
    if state.device != device:
        raise ValueError(f"state device {state.device} does not match token device {device}")
    return state.float()


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[:2] != mask.shape:
        raise ValueError(f"tokens/mask shape mismatch: {tuple(tokens.shape)} vs {tuple(mask.shape)}")
    weights = mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class _FiLMAdaLNResidual(nn.Module):
    """Residual FFN with condition-produced AdaLN/FiLM scale and shift."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.condition = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape != (tokens.shape[0], tokens.shape[-1]):
            raise ValueError(
                f"condition must have shape [{tokens.shape[0]}, {tokens.shape[-1]}], got {tuple(condition.shape)}"
            )
        scale, shift = self.condition(condition.float()).view(tokens.shape[0], 1, 2, tokens.shape[-1]).unbind(dim=2)
        normed = self.norm(tokens.float())
        modulated = normed * (1.0 + scale) + shift
        return tokens.float() + self.ffn(modulated)


class _ConditionEncoder(nn.Module):
    def __init__(self, config: AdvancedConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.language_proj = nn.Linear(config.language_dim, config.hidden_dim)
        self.language_position = nn.Parameter(
            torch.randn(config.max_language_tokens, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_position = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) / math.sqrt(config.hidden_dim)
        )
        self.state_proj = nn.Sequential(nn.LayerNorm(config.state_dim), nn.Linear(config.state_dim, config.hidden_dim))
        self.horizon_embed = nn.Embedding(config.max_horizon, config.hidden_dim)
        self.global_out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def _maybe_drop(
        self,
        language_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob = float(self.config.condition_dropout_prob)
        if not self.training or prob == 0.0:
            return language_tokens, actions, state
        batch_size = language_tokens.shape[0]
        keep_language = (torch.rand(batch_size, 1, 1, device=language_tokens.device) >= prob).to(language_tokens.dtype)
        keep_actions = (torch.rand(batch_size, 1, 1, device=actions.device) >= prob).to(actions.dtype)
        keep_state = (torch.rand(batch_size, 1, device=state.device) >= prob).to(state.dtype)
        return language_tokens * keep_language, actions * keep_actions, state * keep_state

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        language_token_features: torch.Tensor,
        language_token_mask: torch.Tensor | None,
        actions: torch.Tensor,
        state: torch.Tensor,
        horizon_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        language_token_features, language_token_mask = _check_language_tokens(
            language_token_features,
            language_token_mask,
            batch_size=batch_size,
            language_dim=self.config.language_dim,
            max_language_tokens=self.config.max_language_tokens,
            device=device,
        )
        actions = _check_actions(
            actions,
            batch_size=batch_size,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            device=device,
        )
        state = _check_state(state, batch_size=batch_size, state_dim=self.config.state_dim, device=device)
        language_token_features, actions, state = self._maybe_drop(language_token_features, actions, state)

        lang_len = language_token_features.shape[1]
        language_memory = self.language_proj(language_token_features)
        language_memory = language_memory + self.language_position[:lang_len].to(device=device).unsqueeze(0)

        action_memory = self.action_proj(actions)
        action_memory = action_memory + self.action_position.to(device=device).unsqueeze(0)

        ids = _check_horizon_ids(
            horizon_ids,
            batch_size=batch_size,
            max_horizon=self.config.max_horizon,
            device=device,
        )
        if ids is None:
            horizon = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        else:
            horizon = self.horizon_embed(ids).float()
        global_condition = (
            _masked_mean(language_memory, language_token_mask)
            + action_memory.mean(dim=1)
            + self.state_proj(state)
            + horizon
        )
        return (
            language_memory.float(),
            language_token_mask,
            action_memory.float(),
            state,
            self.global_out(global_condition),
        )


class _AdvancedEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.geometry = _DeltaEncoderLayer(hidden_dim, num_heads, dropout)
        self.language = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.action = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.film = _FiLMAdaLNResidual(hidden_dim, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        geometry_memory: torch.Tensor,
        geometry_mask: torch.Tensor | None,
        language_memory: torch.Tensor,
        language_mask: torch.Tensor,
        action_memory: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.geometry(queries, geometry_memory, geometry_mask)
        queries = self.language(queries, language_memory, language_mask)
        queries = self.action(queries, action_memory)
        return self.film(queries, global_condition)


class AdvancedConditionalDeltaEncoder(nn.Module):
    """Encode current/future teacher tokens into condition-grounded delta tokens.

    Shape contract:
    - current_tokens: [B, N, D_teacher]
    - future_tokens: [B, N, D_teacher]
    - language_token_features: [B, L_language, D_language]
    - language_token_mask: optional [B, L_language]
    - actions: [B, H_action, A_dim]
    - state: [B, D_state]
    - output: [B, M_delta, D_delta]
    """

    def __init__(self, config: AdvancedConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = _ConditionEncoder(config)
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
        self.query_condition = nn.Linear(config.hidden_dim, config.num_delta_tokens * config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                _AdvancedEncoderLayer(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.encoder_layers)
            ]
        )
        self.out = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.delta_dim))

    def _horizon_features(
        self, horizon_ids: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        ids = _check_horizon_ids(
            horizon_ids,
            batch_size=batch_size,
            max_horizon=self.config.max_horizon,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, 1, self.config.hidden_dim, device=device)
        return self.horizon_embed(ids).float().unsqueeze(1)

    def _camera_features(
        self,
        camera_ids: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        ids = _check_camera_ids(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            max_cameras=self.config.num_camera_embeddings,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, num_tokens, self.config.hidden_dim, device=device)
        if self.camera_embed is None:
            raise RuntimeError("camera_embed is unexpectedly None")
        return self.camera_embed(ids).float()

    def _build_geometry_memory(
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
        horizon = _expand_to_tokens(self._horizon_features(horizon_ids, batch_size, device), num_tokens)
        position = (
            self.position_embed[:num_tokens]
            .to(device=device)
            .unsqueeze(0)
            .expand(batch_size, num_tokens, self.config.hidden_dim)
        )
        camera = self._camera_features(camera_ids, batch_size=batch_size, num_tokens=num_tokens, device=device)
        current_type = self.current_type.to(device=device).expand(batch_size, num_tokens, self.config.hidden_dim)
        future_type = self.future_type.to(device=device).expand(batch_size, num_tokens, self.config.hidden_dim)
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
        language_token_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        language_token_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _num_tokens, _teacher_dim = _check_teacher_pair(current_tokens, future_tokens)
        language_memory, language_mask, action_memory, _state, global_condition = self.condition(
            batch_size=batch_size,
            device=current_tokens.device,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=actions,
            state=state,
            horizon_ids=horizon_ids,
        )
        geometry_memory, geometry_mask = self._build_geometry_memory(
            current_tokens,
            future_tokens,
            token_mask,
            horizon_ids,
            camera_ids,
        )
        horizon = _expand_to_tokens(
            self._horizon_features(horizon_ids, batch_size, current_tokens.device),
            self.config.num_delta_tokens,
        )
        queries = (
            self.delta_queries.to(device=current_tokens.device)
            .unsqueeze(0)
            .expand(batch_size, self.config.num_delta_tokens, self.config.hidden_dim)
        )
        queries = (
            queries
            + horizon
            + self.query_condition(global_condition).view(
                batch_size, self.config.num_delta_tokens, self.config.hidden_dim
            )
        )
        for layer in self.layers:
            queries = layer(
                queries,
                geometry_memory,
                geometry_mask,
                language_memory,
                language_mask,
                action_memory,
                global_condition,
            )
        return self.out(queries)


class _AdvancedDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.delta = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.language = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.action = _CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.film = _FiLMAdaLNResidual(hidden_dim, dropout)

    def forward(
        self,
        query: torch.Tensor,
        delta_memory: torch.Tensor,
        language_memory: torch.Tensor,
        language_mask: torch.Tensor,
        action_memory: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        query = self.delta(query, delta_memory)
        query = self.language(query, language_memory, language_mask)
        query = self.action(query, action_memory)
        return self.film(query, global_condition)


class AdvancedConditionalDeltaDecoder(nn.Module):
    """Decode future teacher tokens with token-level language/action conditions."""

    def __init__(self, config: AdvancedConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.condition = _ConditionEncoder(config)
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
                _AdvancedDecoderLayer(config.hidden_dim, config.num_heads, config.dropout)
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
        self, horizon_ids: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        ids = _check_horizon_ids(
            horizon_ids,
            batch_size=batch_size,
            max_horizon=self.config.max_horizon,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, 1, self.config.hidden_dim, device=device)
        return self.horizon_embed(ids).float().unsqueeze(1)

    def _camera_features(
        self,
        camera_ids: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        ids = _check_camera_ids(
            camera_ids,
            batch_size=batch_size,
            num_tokens=num_tokens,
            max_cameras=self.config.num_camera_embeddings,
            device=device,
        )
        if ids is None:
            return torch.zeros(batch_size, num_tokens, self.config.hidden_dim, device=device)
        if self.camera_embed is None:
            raise RuntimeError("camera_embed is unexpectedly None")
        return self.camera_embed(ids).float()

    def forward(
        self,
        current_tokens: torch.Tensor,
        delta_tokens: torch.Tensor,
        *,
        language_token_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        language_token_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _assert_rank3("current_tokens", current_tokens)
        _assert_rank3("delta_tokens", delta_tokens)
        batch_size, num_tokens, teacher_dim = current_tokens.shape
        if teacher_dim != self.config.teacher_dim:
            raise ValueError(f"teacher dim {teacher_dim} does not match config.teacher_dim={self.config.teacher_dim}")
        if delta_tokens.shape[:1] != (batch_size,) or delta_tokens.shape[-1] != self.config.delta_dim:
            raise ValueError(
                f"delta_tokens must have shape [B, M, {self.config.delta_dim}], got {tuple(delta_tokens.shape)}"
            )
        if num_tokens > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.config.max_tokens}")
        device = current_tokens.device
        _check_token_mask(token_mask, batch_size=batch_size, num_tokens=num_tokens, device=device)
        language_memory, language_mask, action_memory, _state, global_condition = self.condition(
            batch_size=batch_size,
            device=device,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=actions,
            state=state,
            horizon_ids=horizon_ids,
        )
        horizon = self._horizon_features(horizon_ids, batch_size, device)
        query_horizon = _expand_to_tokens(horizon, num_tokens)
        delta_horizon = _expand_to_tokens(horizon, delta_tokens.shape[1])
        position = (
            self.position_embed[:num_tokens]
            .to(device=device)
            .unsqueeze(0)
            .expand(batch_size, num_tokens, self.config.hidden_dim)
        )
        camera = self._camera_features(camera_ids, batch_size=batch_size, num_tokens=num_tokens, device=device)
        query = self.current_proj(current_tokens.float()) + position + query_horizon + camera
        delta_memory = self.delta_proj(delta_tokens.float()) + delta_horizon
        for layer in self.layers:
            query = layer(query, delta_memory, language_memory, language_mask, action_memory, global_condition)
        return current_tokens.float() + self.residual_out(query.float())


class AdvancedConditionalDeltaTokenizer(nn.Module):
    """Encoder/decoder wrapper for Route B advanced conditional delta tokens."""

    def __init__(self, config: AdvancedConditionalDeltaTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AdvancedConditionalDeltaEncoder(config)
        self.decoder = AdvancedConditionalDeltaDecoder(config)

    def encode(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        language_token_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        language_token_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(
            current_tokens,
            future_tokens,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
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
        language_token_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        language_token_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            current_tokens,
            delta_tokens,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
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
        language_token_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        language_token_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        horizon_ids: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> DeltaTokenizerOutput:
        delta_tokens = self.encode(
            current_tokens,
            future_tokens,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        future_recon = self.decode(
            current_tokens,
            delta_tokens,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        return DeltaTokenizerOutput(delta_tokens=delta_tokens, future_recon=future_recon)

    def config_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self.config)


def advanced_conditional_delta_tokenizer_loss(
    tokenizer: AdvancedConditionalDeltaTokenizer,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    language_token_features: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    language_token_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    horizon_ids: torch.Tensor | None = None,
    camera_ids: torch.Tensor | None = None,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    relation_weight: float = 0.05,
    identity_weight: float = 0.0,
    negative_condition_weight: float = 0.0,
    negative_condition_margin: float = 0.05,
    negative_condition_modes: tuple[str, ...] = ("language", "action"),
) -> DeltaTokenizerLossOutput:
    output = tokenizer(
        current_tokens,
        future_tokens,
        language_token_features=language_token_features,
        language_token_mask=language_token_mask,
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
        "advanced_conditional_delta_tokenizer_loss": rec_loss.detach(),
        "advanced_conditional_delta_rec_loss": rec_loss.detach(),
        "advanced_conditional_delta_abs_mean": output.delta_tokens.float().abs().mean().detach(),
        **{f"advanced_conditional_delta_{key}": value.detach() for key, value in rec_metrics.items()},
    }

    if identity_weight != 0.0:
        zero_delta = torch.zeros_like(output.delta_tokens)
        identity_recon = tokenizer.decode(
            current_tokens,
            zero_delta,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        identity_loss, _ = geometry_distance(
            identity_recon,
            current_tokens,
            token_mask,
            smooth_l1_weight=smooth_l1_weight,
            cosine_weight=cosine_weight,
            relation_weight=0.0,
            relation_max_tokens=tokenizer.config.relation_max_tokens,
        )
        loss = loss + float(identity_weight) * identity_loss
        metrics["advanced_conditional_delta_identity_recon_loss"] = identity_loss.detach()

    neg_losses: list[torch.Tensor] = []
    invalid_modes = set(negative_condition_modes) - {"language", "action"}
    if invalid_modes:
        raise ValueError(f"Unsupported negative_condition_modes: {sorted(invalid_modes)}")
    if negative_condition_weight != 0.0 and "language" in negative_condition_modes:
        neg_language, valid = roll_batch_condition(language_token_features)
        neg_language_mask = language_token_mask
        if language_token_mask is not None:
            neg_language_mask, mask_valid = roll_batch_condition(language_token_mask)
            valid = valid & mask_valid
        neg_output = tokenizer(
            current_tokens,
            future_tokens,
            language_token_features=neg_language,
            language_token_mask=neg_language_mask,
            actions=actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        neg_rec, _ = geometry_distance(
            neg_output.future_recon,
            future_tokens,
            token_mask,
            smooth_l1_weight=smooth_l1_weight,
            cosine_weight=cosine_weight,
            relation_weight=relation_weight,
            relation_max_tokens=tokenizer.config.relation_max_tokens,
        )
        neg_loss = margin_negative_condition_loss(
            rec_loss,
            neg_rec,
            margin=negative_condition_margin,
            valid=valid,
        )
        neg_losses.append(neg_loss)
        metrics["advanced_conditional_neg_language_loss"] = neg_loss.detach()
        metrics["advanced_conditional_neg_language_rec_loss"] = neg_rec.detach()
        metrics["advanced_conditional_neg_language_valid"] = valid.to(dtype=torch.float32).detach()
    if negative_condition_weight != 0.0 and "action" in negative_condition_modes:
        neg_actions, valid = roll_batch_condition(actions)
        neg_output = tokenizer(
            current_tokens,
            future_tokens,
            language_token_features=language_token_features,
            language_token_mask=language_token_mask,
            actions=neg_actions,
            state=state,
            token_mask=token_mask,
            horizon_ids=horizon_ids,
            camera_ids=camera_ids,
        )
        neg_rec, _ = geometry_distance(
            neg_output.future_recon,
            future_tokens,
            token_mask,
            smooth_l1_weight=smooth_l1_weight,
            cosine_weight=cosine_weight,
            relation_weight=relation_weight,
            relation_max_tokens=tokenizer.config.relation_max_tokens,
        )
        neg_loss = margin_negative_condition_loss(
            rec_loss,
            neg_rec,
            margin=negative_condition_margin,
            valid=valid,
        )
        neg_losses.append(neg_loss)
        metrics["advanced_conditional_neg_action_loss"] = neg_loss.detach()
        metrics["advanced_conditional_neg_action_rec_loss"] = neg_rec.detach()
        metrics["advanced_conditional_neg_action_valid"] = valid.to(dtype=torch.float32).detach()
    if neg_losses:
        negative_loss = torch.stack(neg_losses).mean()
        loss = loss + float(negative_condition_weight) * negative_loss
        metrics["advanced_conditional_negative_condition_loss"] = negative_loss.detach()

    return DeltaTokenizerLossOutput(loss=loss, metrics=metrics, model_output=output)
