"""Scheme4 grounded LAC variant using a conditional teacher DeltaTokenizer."""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.conditional_delta_tokenizer import ConditionalDeltaTokenizer
from openpi.models_pytorch.conditional_delta_tokenizer import ConditionalDeltaTokenizerConfig
from openpi.models_pytorch.pi0_align_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme4LACDoubleAlignPI0Pytorch


class Scheme4GroundedDecoderActionBridge(nn.Module):
    """Map deployable action intent to the conditional decoder action shape.

    Shape contract:
    - action_intent: [B, D_intent]
    - output actions: [B, H_action, A_dim]

    The bridge is train-time only for predicted future reconstruction.  It is
    zero-initialized so the new decoder condition starts from an explicit zero
    action chunk rather than an arbitrary projection.
    """

    def __init__(self, *, action_intent_dim: int, action_horizon: int, action_dim: int) -> None:
        super().__init__()
        self.action_intent_dim = int(action_intent_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        if self.action_intent_dim <= 0 or self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_intent_dim, action_horizon, and action_dim must be positive")
        self.proj = nn.Sequential(
            nn.LayerNorm(self.action_intent_dim),
            nn.Linear(self.action_intent_dim, self.action_horizon * self.action_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, action_intent: torch.Tensor) -> torch.Tensor:
        if action_intent.shape[-1:] != (self.action_intent_dim,) or action_intent.ndim != 2:
            raise ValueError(
                f"action_intent must have shape [B, {self.action_intent_dim}], got {tuple(action_intent.shape)}"
            )
        batch_size = int(action_intent.shape[0])
        flat = self.proj(action_intent.float())
        return flat.view(batch_size, self.action_horizon, self.action_dim)


class Scheme4GroundedLACDoubleAlignPI0Pytorch(Scheme4LACDoubleAlignPI0Pytorch):
    """Scheme4 with language/action conditioned teacher delta targets.

    Compared with ``scheme4_lac_double_align``, this variant changes only the
    training-side tokenizer target and decoder path.  Inference still uses the
    inherited teacher-free scheme4 path and keeps the same prefix KV-cache
    behavior.
    """

    SUPPORTED_DECODER_ACTION_SOURCES = ("student_intent", "gt_action")

    def __init__(self, config, extra_config):
        super().__init__(config, extra_config)
        tokenizer_config = self.scheme4_tokenizer.config
        if tokenizer_config.language_dim != self.LLM_width:
            raise ValueError(
                "conditional tokenizer language_dim must match pi language width "
                f"{self.LLM_width}, got {tokenizer_config.language_dim}"
            )
        if tokenizer_config.action_horizon != self.config.action_horizon:
            raise ValueError(
                "conditional tokenizer action_horizon must match model action_horizon "
                f"{self.config.action_horizon}, got {tokenizer_config.action_horizon}"
            )
        if tokenizer_config.action_dim != self.config.action_dim:
            raise ValueError(
                f"conditional tokenizer action_dim must match model action_dim {self.config.action_dim}, "
                f"got {tokenizer_config.action_dim}"
            )
        if tokenizer_config.state_dim != self.config.action_dim:
            raise ValueError(
                f"conditional tokenizer state_dim must match normalized state dim {self.config.action_dim}, "
                f"got {tokenizer_config.state_dim}"
            )
        self.scheme4_grounded_decoder_action_source = str(
            getattr(extra_config, "scheme4_grounded_decoder_action_source", "student_intent")
        )
        if self.scheme4_grounded_decoder_action_source not in self.SUPPORTED_DECODER_ACTION_SOURCES:
            raise ValueError(
                "scheme4_grounded_decoder_action_source must be one of "
                f"{self.SUPPORTED_DECODER_ACTION_SOURCES}, got {self.scheme4_grounded_decoder_action_source!r}"
            )
        self.scheme4_grounded_action_bridge = Scheme4GroundedDecoderActionBridge(
            action_intent_dim=self.action_out_proj.in_features,
            action_horizon=tokenizer_config.action_horizon,
            action_dim=tokenizer_config.action_dim,
        )
        if self.scheme4_grounded_decoder_action_source != "student_intent":
            for param in self.scheme4_grounded_action_bridge.parameters():
                param.requires_grad_(False)

    def _load_scheme4_tokenizer(self, checkpoint_path: str | None) -> ConditionalDeltaTokenizer:
        if checkpoint_path is None:
            raise ValueError("scheme4_grounded_lac_double_align requires extra_config.scheme4_tokenizer_checkpoint")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Conditional DeltaTokenizer checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
            raise ValueError(f"Invalid ConditionalDeltaTokenizer checkpoint payload at {checkpoint_path}")
        if "language_dim" not in payload["config"] or "action_horizon" not in payload["config"]:
            raise ValueError(
                "scheme4_grounded_lac_double_align requires a conditional tokenizer checkpoint, "
                f"but {checkpoint_path} looks like an unconditioned DeltaTokenizer"
            )
        tokenizer = ConditionalDeltaTokenizer(ConditionalDeltaTokenizerConfig(**payload["config"]))
        tokenizer.load_state_dict(payload["state_dict"])
        tokenizer.eval()
        return tokenizer

    def _scheme4_gt_action_condition(self, actions: torch.Tensor) -> torch.Tensor:
        expected = (actions.shape[0], self.config.action_horizon, self.config.action_dim)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")
        return actions.float().detach()

    def _scheme4_decoder_action_condition(self, actions: torch.Tensor, action_intent: torch.Tensor) -> torch.Tensor:
        if self.scheme4_grounded_decoder_action_source == "gt_action":
            return self._scheme4_gt_action_condition(actions)
        return self.scheme4_grounded_action_bridge(action_intent.detach())

    def _scheme4_teacher_delta_targets_grounded(
        self,
        teacher_current_hidden: torch.Tensor,
        teacher_future_hidden: torch.Tensor,
        vision_mask: torch.Tensor | None,
        *,
        language_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_ids = self._scheme4_horizon_ids(teacher_current_hidden.shape[0], teacher_current_hidden.device)
        gt_action_condition = self._scheme4_gt_action_condition(actions)
        with torch.no_grad():
            target_delta_tokens = self.scheme4_tokenizer.encode(
                teacher_current_hidden.detach(),
                teacher_future_hidden.detach(),
                language_features=language_features.detach(),
                actions=gt_action_condition,
                state=state.float().detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
            oracle_future_recon = self.scheme4_tokenizer.decode(
                teacher_current_hidden.detach(),
                target_delta_tokens.detach(),
                language_features=language_features.detach(),
                actions=gt_action_condition,
                state=state.float().detach(),
                token_mask=vision_mask,
                horizon_ids=horizon_ids,
            )
        return target_delta_tokens.detach(), oracle_future_recon.detach()

    def _scheme4_decode_future_grounded(
        self,
        spatial_tokens: torch.Tensor,
        pred_delta_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None,
        *,
        language_features: torch.Tensor,
        decoder_actions: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        horizon_ids = self._scheme4_horizon_ids(spatial_tokens.shape[0], spatial_tokens.device)
        return self.scheme4_tokenizer.decode(
            spatial_tokens,
            pred_delta_tokens,
            language_features=language_features.float(),
            actions=decoder_actions.float(),
            state=state.float(),
            token_mask=vision_mask,
            horizon_ids=horizon_ids,
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
        action_intent = self._scheme4_action_intent_features(suffix_out)
        pred_delta_tokens = self._scheme4_predict_delta(
            spatial_tokens,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
        )
        base_v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), suffix_out)
        v_t, pred_delta_tokens, delta_residual = self._apply_scheme4_delta_fusion(
            suffix_out,
            spatial_tokens,
            vision_mask=vision_mask,
            language_features=language_features,
            state=state,
            action_intent=action_intent,
            pred_delta_tokens=pred_delta_tokens,
        )
        flow_loss = F.mse_loss(u_t, v_t)
        base_flow_loss = F.mse_loss(u_t, base_v_t.float()).detach()

        zero = spatial_tokens.float().sum() * 0.0
        delta_token_loss = zero
        future_recon_loss = zero
        delta_token_metrics: dict[str, torch.Tensor] = {}
        future_metrics: dict[str, torch.Tensor] = {}
        current_as_future_loss = zero.detach()
        oracle_future_loss = zero.detach()
        teacher_future_hidden = None
        decoder_actions = self._scheme4_decoder_action_condition(actions, action_intent)
        needs_future_teacher = self.scheme4_delta_token_loss_coeff != 0.0 or self.scheme4_future_recon_loss_coeff != 0.0
        if self.scheme4_delta_enabled and needs_future_teacher:
            teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)
        if (
            self.scheme4_delta_enabled
            and self.scheme4_delta_token_loss_coeff != 0.0
            and teacher_future_hidden is not None
        ):
            teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
            target_delta_tokens, oracle_future_recon = self._scheme4_teacher_delta_targets_grounded(
                teacher_current_hidden,
                teacher_future_hidden,
                vision_mask,
                language_features=language_features,
                actions=actions,
                state=state,
            )
            delta_token_loss, delta_token_metrics = self._scheme4_delta_token_loss(
                pred_delta_tokens,
                target_delta_tokens,
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
            pred_future_tokens = self._scheme4_decode_future_grounded(
                spatial_tokens,
                pred_delta_tokens,
                vision_mask,
                language_features=language_features,
                decoder_actions=decoder_actions,
                state=state,
            )
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
        )
        spatial_gate = self.scheme2_adapter.gate.float()
        output = {
            "loss": loss,
            "flow_loss": flow_loss,
            "base_flow_loss": base_flow_loss,
            "current_geo_loss": current_geo_loss,
            "future_recon_loss": future_recon_loss,
            "delta_token_loss": delta_token_loss,
            "spatial_gate_abs_mean": spatial_gate.abs().mean().detach(),
            "spatial_gate_max_abs": spatial_gate.abs().max().detach(),
            "delta_gate_abs_mean": self.scheme4_delta_adapter.gate_abs_mean().detach(),
            "delta_gate_max_abs": self.scheme4_delta_adapter.gate_max_abs().detach(),
            "delta_token_norm": pred_delta_tokens.float().norm(dim=-1).mean().detach(),
            "delta_residual_norm": delta_residual.float().norm(dim=-1).mean().detach(),
            "language_summary_norm": language_features.float().norm(dim=-1).mean().detach(),
            "action_intent_norm": action_intent.float().norm(dim=-1).mean().detach(),
            "language_token_residual_norm": torch.tensor(0.0, dtype=torch.float32, device=loss.device),
            "scheme4_delta_enabled": torch.tensor(
                float(self.scheme4_delta_enabled), dtype=torch.float32, device=loss.device
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
            "scheme4_grounded_gt_action_norm": actions.float().norm(dim=-1).mean().detach(),
            "scheme4_grounded_decoder_action_norm": decoder_actions.float().norm(dim=-1).mean().detach(),
            "scheme4_grounded_decoder_uses_student_intent": torch.tensor(
                float(self.scheme4_grounded_decoder_action_source == "student_intent"),
                dtype=torch.float32,
                device=loss.device,
            ),
            "scheme2_unified_spatial_loss_skipped": spatial_loss_skipped.detach(),
        }
        output.update(spatial_metrics)
        output.update(delta_token_metrics)
        output.update(future_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
