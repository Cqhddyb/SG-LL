import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    loss = 1.0 - (pred * target).sum(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.bool()
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def geometry_delta_loss(
    pred_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    valid_batch_mask: torch.Tensor | None = None,
    *,
    norm_coeff: float = 0.05,
    global_coeff: float = 0.2,
    motion_temp: float = 0.5,
    min_teacher_norm: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_delta = teacher_delta.detach()
    bsz, num_tokens, _ = teacher_delta.shape
    device = teacher_delta.device

    if token_mask is None:
        token_mask = torch.ones(bsz, num_tokens, device=device, dtype=torch.bool)
    else:
        token_mask = token_mask.bool()

    if valid_batch_mask is None:
        valid_batch_mask = torch.ones(bsz, device=device, dtype=torch.bool)
    else:
        valid_batch_mask = valid_batch_mask.bool()

    eps = 1e-6
    teacher_token_norm = teacher_delta.float().norm(dim=-1)
    masked_norm = teacher_token_norm * token_mask.float()
    mean_norm = masked_norm.sum(dim=1, keepdim=True) / token_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    score = masked_norm / (mean_norm + eps)
    score = score.masked_fill(~token_mask, -1e4)
    motion_weight = torch.softmax(score / max(float(motion_temp), 1e-3), dim=1).detach()
    motion_weight = motion_weight * token_mask.float()
    motion_weight = motion_weight / motion_weight.sum(dim=1, keepdim=True).clamp_min(eps)

    teacher_global = (teacher_delta.float() * motion_weight.unsqueeze(-1)).sum(dim=1)
    teacher_global_norm = teacher_global.norm(dim=-1)
    if float(min_teacher_norm) > 0:
        valid_batch_mask = valid_batch_mask & (teacher_global_norm > float(min_teacher_norm))

    if not valid_batch_mask.any():
        zero = pred_delta.float().sum() * 0.0
        metrics = {
            "delta_main_loss": zero.detach(),
            "delta_norm_loss": zero.detach(),
            "delta_global_cos_loss": zero.detach(),
            "delta_valid_ratio": valid_batch_mask.float().mean().detach(),
            "teacher_delta_norm_valid": zero.detach(),
            "pred_delta_norm_valid": zero.detach(),
        }
        return zero, metrics

    pred_valid = pred_delta[valid_batch_mask].float()
    teacher_valid = teacher_delta[valid_batch_mask].float()
    weight_valid = motion_weight[valid_batch_mask].float()

    token_cos = F.cosine_similarity(pred_valid, teacher_valid, dim=-1, eps=1e-6)
    token_cos_loss = ((1.0 - token_cos) * weight_valid).sum(dim=1).mean()

    pred_token_norm = pred_valid.norm(dim=-1)
    teacher_token_norm_valid = teacher_valid.norm(dim=-1).detach()
    token_norm_loss_all = F.smooth_l1_loss(pred_token_norm, teacher_token_norm_valid, reduction="none")
    norm_loss = (token_norm_loss_all * weight_valid).sum(dim=1).mean()

    pred_global = (pred_valid * weight_valid.unsqueeze(-1)).sum(dim=1)
    teacher_global_valid = (teacher_valid * weight_valid.unsqueeze(-1)).sum(dim=1)
    global_cos_loss = 1.0 - F.cosine_similarity(pred_global, teacher_global_valid, dim=-1, eps=1e-6).mean()

    loss = token_cos_loss + float(norm_coeff) * norm_loss + float(global_coeff) * global_cos_loss
    metrics = {
        "delta_main_loss": token_cos_loss.detach(),
        "delta_norm_loss": norm_loss.detach(),
        "delta_global_cos_loss": global_cos_loss.detach(),
        "delta_valid_ratio": valid_batch_mask.float().mean().detach(),
        "teacher_delta_norm_valid": teacher_global_norm[valid_batch_mask].mean().detach(),
        "pred_delta_norm_valid": pred_global.norm(dim=-1).mean().detach(),
    }
    return loss, metrics


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding_mask = None if memory_mask is None else ~memory_mask.bool()
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attn_out
        return query + self.ffn(query)


class GeoAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_slots: int = 16,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim) / math.sqrt(hidden_dim))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.input_proj(hidden)
        slots = self.slots.unsqueeze(0).expand(hidden.shape[0], -1, -1).to(dtype=memory.dtype, device=memory.device)
        return self.out_norm(self.cross_attn(slots, memory, mask))


class TeacherTokenDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        max_tokens: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(max_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, slots: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if num_tokens > self.target_queries.shape[0]:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.target_queries.shape[0]}")
        queries = self.target_queries[:num_tokens].unsqueeze(0).expand(slots.shape[0], -1, -1)
        queries = queries.to(dtype=slots.dtype, device=slots.device)
        decoded = self.cross_attn(queries, slots)
        return self.out(decoded)


class GeometryVariationHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        max_tokens: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(max_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.action_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, z_geo: torch.Tensor, z_act: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if num_tokens > self.target_queries.shape[0]:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.target_queries.shape[0]}")
        queries = self.target_queries[:num_tokens].unsqueeze(0).expand(z_geo.shape[0], -1, -1)
        queries = queries.to(dtype=z_geo.dtype, device=z_geo.device)

        memory = torch.cat([z_geo, z_act], dim=1)
        fused = self.cross_attn(queries, memory)
        act_summary = z_act.mean(dim=1)
        gamma, beta = self.action_gate(act_summary).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        fused = fused * (1.0 + gamma) + beta
        combined = torch.cat([queries, fused, queries - fused], dim=-1)
        return self.out(combined)


class ActionIntentAdapter(nn.Module):
    def __init__(
        self,
        action_dim: int,
        prefix_dim: int,
        hidden_dim: int = 512,
        action_horizon: int = 10,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prefix_proj = nn.Linear(prefix_dim, hidden_dim)
        self.horizon_embed = nn.Parameter(torch.randn(action_horizon, hidden_dim) / math.sqrt(hidden_dim))
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        time_embedding: torch.Tensor,
        prefix_hidden: torch.Tensor,
        z_geo: torch.Tensor,
        prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        horizon = noisy_actions.shape[1]
        action_q = self.action_proj(noisy_actions)
        time_q = self.time_proj(time_embedding).unsqueeze(1)
        horizon_q = self.horizon_embed[:horizon].unsqueeze(0).to(dtype=action_q.dtype, device=action_q.device)
        query = action_q + time_q + horizon_q

        prefix_memory = self.prefix_proj(prefix_hidden)
        memory = torch.cat([prefix_memory, z_geo], dim=1)
        if prefix_mask is not None:
            geo_mask = torch.ones(z_geo.shape[:2], dtype=torch.bool, device=z_geo.device)
            memory_mask = torch.cat([prefix_mask.bool(), geo_mask], dim=1)
        else:
            memory_mask = None

        query = self.cross_attn(query, memory, memory_mask)
        query = self.temporal_encoder(query)
        return self.out_norm(query)


class ResidualFlowExpert(nn.Module):
    def __init__(
        self,
        suffix_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.suffix_proj = nn.Linear(suffix_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.geo_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        suffix_out: torch.Tensor,
        noisy_actions: torch.Tensor,
        z_geo: torch.Tensor,
        z_act: torch.Tensor,
    ) -> torch.Tensor:
        query = self.suffix_proj(suffix_out) + self.action_proj(noisy_actions)
        memory = torch.cat([self.geo_proj(z_geo), z_act], dim=1)
        hidden = self.cross_attn(query, memory)
        return self.out(hidden)
