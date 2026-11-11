"""Small utilities for pi0/pi0.5 flow-matching tensors."""

from __future__ import annotations

import torch


def estimate_action_endpoint(
    noisy_actions: torch.Tensor,
    timestep: torch.Tensor,
    velocity: torch.Tensor,
    *,
    detach: bool = True,
) -> torch.Tensor:
    """Estimate the denoised action endpoint from one flow-matching state.

    pi0.5 training uses:
    - x_t = t * noise + (1 - t) * action
    - v_t = noise - action

    Therefore the action endpoint implied by a predicted velocity is:
    - action_hat = x_t - t * v_hat

    Shape contract:
    - noisy_actions: [B, H_action, A_dim]
    - timestep: [B]
    - velocity: [B, H_action, A_dim]
    - output: [B, H_action, A_dim]
    """

    if noisy_actions.ndim != 3:
        raise ValueError(f"noisy_actions must be rank 3, got shape {tuple(noisy_actions.shape)}")
    if velocity.shape != noisy_actions.shape:
        raise ValueError(
            f"velocity shape {tuple(velocity.shape)} must match noisy_actions shape {tuple(noisy_actions.shape)}"
        )
    if timestep.shape != (noisy_actions.shape[0],):
        raise ValueError(f"timestep must have shape {(noisy_actions.shape[0],)}, got {tuple(timestep.shape)}")
    if velocity.device != noisy_actions.device:
        raise ValueError(
            f"velocity device {velocity.device} does not match noisy_actions device {noisy_actions.device}"
        )
    if timestep.device != noisy_actions.device:
        raise ValueError(
            f"timestep device {timestep.device} does not match noisy_actions device {noisy_actions.device}"
        )
    if torch.is_complex(noisy_actions) or torch.is_complex(velocity) or torch.is_complex(timestep):
        raise ValueError("noisy_actions, velocity, and timestep must be real-valued tensors")

    time_expanded = timestep.to(dtype=noisy_actions.dtype).view(noisy_actions.shape[0], 1, 1).expand_as(noisy_actions)
    action_hat = noisy_actions - time_expanded * velocity.to(dtype=noisy_actions.dtype)
    if detach:
        action_hat = action_hat.detach()
    return action_hat
