from __future__ import annotations

import pytest
import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.future_state_tokenizer import FutureStateTokenizer
from openpi.models_pytorch.pi0_proposal_variants_pytorch import get_proposal_model_class
from openpi.models_pytorch.scheme4_frame_token_lac_pytorch import Scheme4FrameTokenLACDoubleAlignPI0Pytorch


def _tokenizer() -> FutureStateTokenizer:
    return FutureStateTokenizer(
        DeltaTokenizerConfig(
            teacher_dim=8,
            hidden_dim=16,
            delta_dim=4,
            num_delta_tokens=2,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            max_tokens=6,
            max_horizon=8,
            relation_max_tokens=6,
            zero_init_decoder=False,
        )
    ).eval()


def _minimal_model() -> Scheme4FrameTokenLACDoubleAlignPI0Pytorch:
    model = object.__new__(Scheme4FrameTokenLACDoubleAlignPI0Pytorch)
    torch.nn.Module.__init__(model)
    model.scheme4_tokenizer = _tokenizer()
    model.scheme4_horizon_id = 2
    return model


def test_frame_token_variant_is_registered_lazily() -> None:
    cls = get_proposal_model_class("scheme4_frame_token_lac_double_align")

    assert cls is Scheme4FrameTokenLACDoubleAlignPI0Pytorch


def test_frame_teacher_target_ignores_current_teacher_content() -> None:
    torch.manual_seed(3)
    model = _minimal_model()
    current_a = torch.randn(2, 5, 8)
    current_b = torch.randn(2, 5, 8) + 30.0
    future = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    tokens_a, recon_a = model._scheme4_teacher_delta_targets(current_a, future, mask)  # noqa: SLF001
    tokens_b, recon_b = model._scheme4_teacher_delta_targets(current_b, future, mask)  # noqa: SLF001

    assert not tokens_a.requires_grad
    assert not recon_a.requires_grad
    assert torch.equal(tokens_a, tokens_b)
    assert torch.equal(recon_a, recon_b)


def test_frame_decoder_ignores_spatial_token_content() -> None:
    torch.manual_seed(5)
    model = _minimal_model()
    spatial_a = torch.randn(2, 5, 8)
    spatial_b = torch.randn(2, 5, 8) - 20.0
    pred_frame = torch.randn(2, 2, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    recon_a = model._scheme4_decode_future(spatial_a, pred_frame, mask)  # noqa: SLF001
    recon_b = model._scheme4_decode_future(spatial_b, pred_frame, mask)  # noqa: SLF001

    assert recon_a.shape == spatial_a.shape
    assert torch.equal(recon_a, recon_b)


def test_frame_variant_rejects_transition_tokenizer_checkpoint(tmp_path) -> None:
    model = _minimal_model()
    transition_path = tmp_path / "transition.pt"
    tokenizer = _tokenizer()
    torch.save(
        {
            "schema_version": "scheme3_delta_tokenizer_cached_checkpoint_v1",
            "config": tokenizer.config_dict(),
            "state_dict": tokenizer.state_dict(),
        },
        transition_path,
    )

    with pytest.raises(ValueError, match="FutureStateTokenizer checkpoint"):
        model._load_scheme4_tokenizer(str(transition_path))  # noqa: SLF001
