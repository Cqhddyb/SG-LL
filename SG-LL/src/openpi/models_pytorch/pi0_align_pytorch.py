import contextlib
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
import openpi.models_pytorch.geo_action_modules as _geo_modules
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from vggt.utils.load_fn import preprocess_images_from_openpi
from vggt.heads.utils import custom_pooling


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config, extra_config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.LLM_width = paligemma_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Specific config for SpatialForcing alignment
        self.vla_layers_align = 12 if extra_config.vla_layers_align is None else extra_config.vla_layers_align
        self.vggt_layers_align = -1 if extra_config.vggt_layers_align is None else extra_config.vggt_layers_align
        self.pooling_func = "bilinear" if extra_config.pooling_func is None else extra_config.pooling_func
        self.use_vggt_pe = True if extra_config.use_vggt_pe is None else extra_config.use_vggt_pe
        self.vggt_dim = int(getattr(extra_config, "vggt_dim", 1024))
        default_teacher_dim = self.vggt_dim
        self.teacher_dim = int(getattr(extra_config, "geo_teacher_dim", default_teacher_dim))
        self.geo_hidden_dim = int(getattr(extra_config, "geo_hidden_dim", 512))
        self.geo_num_slots = int(getattr(extra_config, "geo_num_slots", 16))
        self.geo_num_heads = int(getattr(extra_config, "geo_num_heads", 8))
        self.geo_dropout = float(getattr(extra_config, "geo_dropout", 0.0))
        self.geo_max_tokens = int(getattr(extra_config, "geo_max_tokens", 1024))
        self.current_geo_loss_coeff = float(getattr(extra_config, "current_geo_loss_coeff", 0.5))
        self.delta_loss_coeff = float(getattr(extra_config, "delta_loss_coeff", 0.02))
        self.delta_norm_coeff = float(getattr(extra_config, "delta_norm_coeff", 0.05))
        self.delta_global_coeff = float(getattr(extra_config, "delta_global_coeff", 0.2))
        self.delta_motion_temp = float(getattr(extra_config, "delta_motion_temp", 0.5))
        self.delta_min_teacher_norm = float(getattr(extra_config, "delta_min_teacher_norm", 0.0))
        self.residual_reg_coeff = float(getattr(extra_config, "residual_reg_coeff", 1e-4))

        self.geo_adapter = _geo_modules.GeoAdapter(
            input_dim=paligemma_config.width,
            hidden_dim=self.geo_hidden_dim,
            num_slots=self.geo_num_slots,
            num_heads=self.geo_num_heads,
            dropout=self.geo_dropout,
        )
        self.teacher_token_decoder = _geo_modules.TeacherTokenDecoder(
            hidden_dim=self.geo_hidden_dim,
            target_dim=self.teacher_dim,
            max_tokens=self.geo_max_tokens,
            num_heads=self.geo_num_heads,
            dropout=self.geo_dropout,
        )
        self.action_intent_adapter = _geo_modules.ActionIntentAdapter(
            action_dim=config.action_dim,
            prefix_dim=paligemma_config.width,
            hidden_dim=self.geo_hidden_dim,
            action_horizon=config.action_horizon,
            num_heads=self.geo_num_heads,
            dropout=self.geo_dropout,
        )
        self.geometry_variation_head = _geo_modules.GeometryVariationHead(
            hidden_dim=self.geo_hidden_dim,
            target_dim=self.teacher_dim,
            max_tokens=self.geo_max_tokens,
            num_heads=self.geo_num_heads,
            dropout=self.geo_dropout,
        )
        self.residual_flow_expert = _geo_modules.ResidualFlowExpert(
            suffix_dim=action_expert_config.width,
            action_dim=config.action_dim,
            hidden_dim=self.geo_hidden_dim,
            num_heads=self.geo_num_heads,
            dropout=self.geo_dropout,
        )

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True, get_wo_aug=False):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train, get_wo_aug=get_wo_aug)
        return (
            list(observation.images.values()),
            list(observation.img_wo_aug.values()) if get_wo_aug else None,
            list(observation.teacher_images.values()) if getattr(observation, "teacher_images", None) is not None else None,
            list(observation.future_teacher_images.values())
            if getattr(observation, "future_teacher_images", None) is not None
            else None,
            getattr(observation, "future_teacher_fallback", None),
            list(observation.image_padding_mask.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        img_len = len(att_masks)

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, img_len

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _build_vision_token_mask(self, img_masks, img_padding_mask, img_len, num_images):
        tokens_per_img = img_len // num_images
        img_masks_stack = torch.stack(img_masks, dim=1)
        align_mask = torch.repeat_interleave(img_masks_stack, repeats=tokens_per_img, dim=1)

        img_padding_mask = torch.stack(img_padding_mask, dim=1)
        target_size = img_padding_mask.shape[-1] // 14
        mask_downsampled = F.interpolate(
            img_padding_mask.float(),
            size=(target_size, target_size),
            mode="nearest",
        ).bool().flatten(start_dim=1)
        if align_mask.shape != mask_downsampled.shape:
            raise RuntimeError(
                f"vision mask shape mismatch: align_mask={tuple(align_mask.shape)}, "
                f"padding_mask={tuple(mask_downsampled.shape)}"
            )
        return align_mask & mask_downsampled

    def _encode_vggt_hidden(self, teacher_images, vggt, vision_hidden_ref):
        if teacher_images is None:
            raise ValueError("teacher images are required for geometry supervision")
        img_resize = preprocess_images_from_openpi(teacher_images)
        autocast_ctx = (
            torch.autocast(img_resize.device.type, dtype=torch.bfloat16)
            if img_resize.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx, torch.no_grad():
            vggt_output = vggt(img_resize)
        agg_vggt_hidden = vggt_output["features"][self.vggt_layers_align]
        patch_start_idx = vggt_output["patch_start_idx"]
        original_img = vggt_output["images"]
        vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]

        H, W = original_img.shape[-2:]
        patch_h, patch_w = H // vggt.patch_size, W // vggt.patch_size
        vggt_hidden = custom_pooling(
            vggt_hidden,
            (patch_h, patch_w),
            (H, W),
            vision_hidden_ref.detach(),
            self.pooling_func,
            self.use_vggt_pe,
        )
        return vggt_hidden.detach()

    def _time_embedding_for_geo(self, timestep):
        emb = create_sinusoidal_pos_embedding(
            timestep,
            self.geo_hidden_dim,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        return emb.to(dtype=torch.float32, device=timestep.device)

    def _compute_geo_conditioning(
        self,
        prefix_hidden,
        vision_hidden,
        vision_mask,
        noisy_actions,
        timestep,
        prefix_mask=None,
    ):
        prefix_hidden = prefix_hidden.float()
        vision_hidden = vision_hidden.float()
        z_geo = self.geo_adapter(vision_hidden, vision_mask)
        if prefix_mask is None:
            prefix_mask = torch.ones(prefix_hidden.shape[:2], dtype=torch.bool, device=prefix_hidden.device)
        time_embedding = self._time_embedding_for_geo(timestep)
        z_act = self.action_intent_adapter(
            noisy_actions.float(),
            time_embedding,
            prefix_hidden,
            z_geo,
            prefix_mask=prefix_mask,
        )
        return z_geo, z_act

    def _apply_residual_flow(self, suffix_out, base_v_t, noisy_actions, z_geo, z_act):
        residual = self.residual_flow_expert(suffix_out.float(), noisy_actions.float(), z_geo.float(), z_act.float())
        return base_v_t + residual, residual

    def forward(self, observation, actions, vggt, align_proj=None, noise=None, time=None) -> dict[str, torch.Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
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

        # =================================== VLA action loss ===================================

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

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
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

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        base_v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # =================================== Alignment loss ===================================

        # VLA hidden states
        (prefix_hidden, _) = all_hidden_states[self.vla_layers_align]  # 18 total layers of paligemma
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        teacher_current_hidden = self._encode_vggt_hidden(teacher_images, vggt, vision_hidden)
        teacher_future_hidden = self._encode_vggt_hidden(future_teacher_images, vggt, vision_hidden)

        if teacher_current_hidden.shape[-1] != self.teacher_dim:
            raise RuntimeError(
                f"VGGT teacher dim {teacher_current_hidden.shape[-1]} does not match geo_teacher_dim={self.teacher_dim}."
            )

        z_geo, z_act = self._compute_geo_conditioning(
            prefix_hidden,
            vision_hidden,
            vision_mask,
            x_t,
            time,
            prefix_mask=prefix_pad_masks,
        )

        pred_current_hidden = self.teacher_token_decoder(z_geo, teacher_current_hidden.shape[1])
        current_geo_loss = _geo_modules.masked_cosine_loss(pred_current_hidden, teacher_current_hidden, vision_mask)

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
            "current_geo_loss_coeff": torch.tensor(self.current_geo_loss_coeff, device=loss.device),
            "delta_loss_coeff": torch.tensor(self.delta_loss_coeff, device=loss.device),
        }
        output.update(delta_metrics)
        return output

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

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

        prefix_embs, prefix_pad_masks, prefix_att_masks, img_len = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, prefix_hidden_states = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            output_hidden_states=True,
        )

        if prefix_hidden_states is None:
            raise RuntimeError("prefix hidden states are required for geometry-aware inference")
        prefix_hidden = prefix_hidden_states[self.vla_layers_align].float()
        vision_hidden = prefix_hidden[:, :img_len, :]
        vision_mask = self._build_vision_token_mask(img_masks, img_padding_mask, img_len, len(images))
        z_geo = self.geo_adapter(vision_hidden, vision_mask)
        geo_cache = {
            "prefix_hidden": prefix_hidden,
            "vision_mask": vision_mask,
            "z_geo": z_geo.float(),
        }

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                geo_cache,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        geo_cache,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        base_v_t = self.action_out_proj(suffix_out)

        prefix_hidden = geo_cache["prefix_hidden"].float()
        z_geo = geo_cache["z_geo"].float()
        z_act = self.action_intent_adapter(
            x_t.float(),
            self._time_embedding_for_geo(timestep),
            prefix_hidden,
            z_geo,
            prefix_mask=prefix_pad_masks,
        )
        v_t, _ = self._apply_residual_flow(suffix_out, base_v_t.float(), x_t, z_geo, z_act)
        return v_t
