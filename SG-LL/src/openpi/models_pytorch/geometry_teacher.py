"""Geometry-teacher feature extraction interface for policy supervision.

Teacher implementations are training-side feature providers.  The policy
serving path consumes the learned student state and does not call this module.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable
from typing import Protocol

import torch

from vggt.heads.utils import custom_pooling
from vggt.utils.load_fn import preprocess_images_from_openpi


@dataclass(frozen=True)
class GeometryTeacherContext:
    """Options needed to align teacher patches to the policy visual grid."""

    feature_layer: int
    pooling_func: str
    use_positional_encoding: bool
    expected_feature_dim: int


class GeometryFeatureTeacher(Protocol):
    """Feature extraction contract shared by train-time geometry teachers."""

    name: str

    def extract_features(
        self,
        images: list[torch.Tensor],
        *,
        teacher_model: torch.nn.Module,
        reference_tokens: torch.Tensor,
        context: GeometryTeacherContext,
    ) -> torch.Tensor:
        """Return detached features aligned to ``reference_tokens`` as ``[B,N,D]``."""


class VGGTGeometryTeacher:
    """Adapter around the repository's existing frozen VGGT implementation."""

    name = "vggt"

    def extract_features(
        self,
        images: list[torch.Tensor],
        *,
        teacher_model: torch.nn.Module,
        reference_tokens: torch.Tensor,
        context: GeometryTeacherContext,
    ) -> torch.Tensor:
        if reference_tokens.ndim != 3:
            raise ValueError(f"reference_tokens must be [B,N,D], got {tuple(reference_tokens.shape)}")
        if not images:
            raise ValueError("images must contain at least one camera view")

        img_resize = preprocess_images_from_openpi(images)
        autocast_ctx = (
            torch.autocast(img_resize.device.type, dtype=torch.bfloat16)
            if img_resize.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx, torch.no_grad():
            vggt_output = teacher_model(img_resize)

        try:
            hidden = vggt_output["features"][context.feature_layer]
            patch_start_idx = int(vggt_output["patch_start_idx"])
            original_images = vggt_output["images"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("VGGT output is missing aligned features, patch_start_idx, or images") from error
        if hidden.ndim != 4:
            raise ValueError(f"VGGT layer features must be [B,C,T,D], got {tuple(hidden.shape)}")
        if patch_start_idx < 0 or patch_start_idx >= hidden.shape[2]:
            raise ValueError(f"invalid VGGT patch_start_idx={patch_start_idx} for token count {hidden.shape[2]}")

        patch_tokens = hidden[:, :, patch_start_idx:, :]
        height, width = original_images.shape[-2:]
        patch_grid = (height // int(teacher_model.patch_size), width // int(teacher_model.patch_size))
        aligned = custom_pooling(
            patch_tokens,
            patch_grid,
            (height, width),
            reference_tokens.detach(),
            context.pooling_func,
            context.use_positional_encoding,
        )
        expected_shape = (reference_tokens.shape[0], reference_tokens.shape[1], context.expected_feature_dim)
        if tuple(aligned.shape) != expected_shape:
            raise ValueError(f"aligned teacher features have shape {tuple(aligned.shape)}, expected {expected_shape}")
        return aligned.detach()


_GEOMETRY_TEACHERS: dict[str, Callable[[], GeometryFeatureTeacher]] = {
    "vggt": VGGTGeometryTeacher,
}


def register_geometry_teacher(name: str, factory: Callable[[], GeometryFeatureTeacher]) -> None:
    """Register a teacher adapter without changing the policy model call sites."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("teacher name must be non-empty")
    if normalized in _GEOMETRY_TEACHERS:
        raise ValueError(f"geometry teacher {normalized!r} is already registered")
    _GEOMETRY_TEACHERS[normalized] = factory


def get_geometry_teacher(name: str) -> GeometryFeatureTeacher:
    """Construct a registered geometry-teacher adapter."""

    normalized = str(name).strip().lower()
    try:
        return _GEOMETRY_TEACHERS[normalized]()
    except KeyError as error:
        supported = ", ".join(sorted(_GEOMETRY_TEACHERS))
        raise ValueError(
            f"unsupported geometry teacher {name!r}; registered adapters: {supported}. "
            "VGGT-Ω requires a separately validated adapter and checkpoint."
        ) from error
