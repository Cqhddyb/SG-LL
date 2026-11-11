"""Offline utilities for building fixed Scheme4 mean-delta controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class DeltaTokenMeanAccumulator:
    """Accumulate detached ``[B, M_delta, D_delta]`` tokens on CPU."""

    def __init__(self, expected_token_shape: tuple[int, int] | None = None):
        self.expected_token_shape = expected_token_shape
        self._sum: torch.Tensor | None = None
        self._sum_sq: torch.Tensor | None = None
        self._norm_sum: torch.Tensor | None = None
        self._norm_sum_sq: torch.Tensor | None = None
        self.count = 0

    def update(self, tokens: torch.Tensor) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}")
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, M_delta, D_delta], got {tuple(tokens.shape)}")
        token_shape = (int(tokens.shape[1]), int(tokens.shape[2]))
        if self.expected_token_shape is not None and token_shape != self.expected_token_shape:
            raise ValueError(f"token shape {token_shape} must match expected {self.expected_token_shape}")
        if tokens.shape[0] <= 0:
            raise ValueError("tokens batch must be non-empty")

        detached = tokens.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(detached).all():
            raise ValueError("tokens contain non-finite values")
        batch_sum = detached.sum(dim=0)
        batch_sum_sq = detached.square().sum(dim=0)
        norms = detached.norm(dim=-1)
        norm_sum = norms.sum(dim=0)
        norm_sum_sq = norms.square().sum(dim=0)
        if self._sum is None:
            self._sum = torch.zeros_like(batch_sum)
            self._sum_sq = torch.zeros_like(batch_sum_sq)
            self._norm_sum = torch.zeros_like(norm_sum)
            self._norm_sum_sq = torch.zeros_like(norm_sum_sq)
        self._sum += batch_sum
        self._sum_sq += batch_sum_sq
        self._norm_sum += norm_sum
        self._norm_sum_sq += norm_sum_sq
        self.count += int(detached.shape[0])

    def finalize(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.count == 0 or self._sum is None or self._sum_sq is None:
            raise ValueError("cannot finalize an empty accumulator")
        assert self._norm_sum is not None
        assert self._norm_sum_sq is not None
        mean64 = self._sum / self.count
        variance = (self._sum_sq / self.count - mean64.square()).clamp_min(0.0)
        norm_mean = self._norm_sum / self.count
        norm_std = (self._norm_sum_sq / self.count - norm_mean.square()).clamp_min(0.0).sqrt()
        mean = mean64.to(dtype=torch.float32)
        stats = {
            "sample_count": self.count,
            "token_shape": list(mean.shape),
            "mean_abs": float(mean.abs().mean()),
            "mean_l2_norm": float(mean.norm(dim=-1).mean()),
            "prediction_element_std_mean": float(variance.sqrt().mean()),
            "prediction_token_l2_norm_mean": [float(value) for value in norm_mean],
            "prediction_token_l2_norm_std": [float(value) for value in norm_std],
        }
        return mean, stats


def deterministic_subset_indices(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if sample_count <= 0 or sample_count > dataset_size:
        raise ValueError(f"sample_count must be in [1, {dataset_size}], got {sample_count}")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(dataset_size, size=sample_count, replace=False))
