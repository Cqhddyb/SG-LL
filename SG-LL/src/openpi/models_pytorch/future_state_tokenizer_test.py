import io

import pytest
import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizer
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.future_state_tokenizer import FutureStateTokenizer
from openpi.models_pytorch.future_state_tokenizer import future_state_tokenizer_loss


def _config(*, zero_init_decoder: bool = True) -> DeltaTokenizerConfig:
    return DeltaTokenizerConfig(
        teacher_dim=16,
        hidden_dim=32,
        delta_dim=8,
        num_delta_tokens=4,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        dropout=0.0,
        max_tokens=12,
        max_horizon=8,
        num_camera_embeddings=3,
        relation_max_tokens=8,
        zero_init_decoder=zero_init_decoder,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4)
    future = torch.randn(2, 6, 16)
    token_mask = torch.tensor([[True, True, True, True, True, False], [True, True, True, False, False, False]])
    horizon_ids = torch.tensor([1, 3], dtype=torch.long)
    camera_ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    return future, token_mask, horizon_ids, camera_ids


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


def test_future_state_tokenizer_shapes_and_masks() -> None:
    tokenizer = FutureStateTokenizer(_config())
    future, token_mask, horizon_ids, camera_ids = _inputs()

    output = tokenizer(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert output.frame_tokens.shape == (2, 4, 8)
    assert output.future_recon.shape == future.shape
    assert output.future_recon.dtype == torch.float32


def test_future_state_tokenizer_parameter_count_matches_transition_tokenizer() -> None:
    config = _config()
    frame_tokenizer = FutureStateTokenizer(config)
    transition_tokenizer = DeltaTokenizer(config)

    frame_parameters = sum(parameter.numel() for parameter in frame_tokenizer.parameters())
    transition_parameters = sum(parameter.numel() for parameter in transition_tokenizer.parameters())

    assert frame_parameters == transition_parameters


def test_future_state_tokenizer_rejects_bad_shapes() -> None:
    tokenizer = FutureStateTokenizer(_config())
    future, token_mask, horizon_ids, camera_ids = _inputs()

    with pytest.raises(ValueError, match="future_tokens"):
        tokenizer(future[:, :, :-1], token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)
    with pytest.raises(ValueError, match="token_mask"):
        tokenizer(future, token_mask=token_mask[:, :-1], horizon_ids=horizon_ids, camera_ids=camera_ids)
    with pytest.raises(ValueError, match="horizon_ids"):
        tokenizer(future, token_mask=token_mask, horizon_ids=horizon_ids[:, None], camera_ids=camera_ids)
    with pytest.raises(ValueError, match="frame_tokens"):
        tokenizer.decode(torch.randn(2, 3, 8), num_tokens=future.shape[1])
    with pytest.raises(ValueError, match="num_tokens"):
        tokenizer.decode(torch.randn(2, 4, 8), num_tokens=0)


def test_future_state_zero_frame_decodes_to_zero_at_zero_init() -> None:
    tokenizer = FutureStateTokenizer(_config(zero_init_decoder=True))
    future, token_mask, horizon_ids, camera_ids = _inputs()
    zero_frame = torch.zeros(2, 4, 8)

    recon = tokenizer.decode(
        zero_frame,
        num_tokens=future.shape[1],
        token_mask=token_mask,
        horizon_ids=horizon_ids,
        camera_ids=camera_ids,
    )

    assert torch.allclose(recon, torch.zeros_like(recon), atol=0.0, rtol=0.0)


def test_future_state_tokenizer_gradient_flow() -> None:
    tokenizer = FutureStateTokenizer(_config(zero_init_decoder=False))
    future, token_mask, horizon_ids, camera_ids = _inputs()
    future = future.requires_grad_()

    output = tokenizer(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)
    loss = output.frame_tokens.square().mean() + output.future_recon.square().mean()
    loss.backward()

    assert future.grad is not None
    assert torch.isfinite(future.grad).all()
    assert _has_nonzero_grad(tokenizer.encoder)
    assert _has_nonzero_grad(tokenizer.decoder)


def test_future_state_tokenizer_deterministic_eval() -> None:
    future, token_mask, horizon_ids, camera_ids = _inputs()
    torch.manual_seed(7)
    tokenizer_a = FutureStateTokenizer(_config(zero_init_decoder=False)).eval()
    torch.manual_seed(7)
    tokenizer_b = FutureStateTokenizer(_config(zero_init_decoder=False)).eval()

    with torch.no_grad():
        output_a = tokenizer_a(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)
        output_b = tokenizer_b(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert torch.equal(output_a.frame_tokens, output_b.frame_tokens)
    assert torch.equal(output_a.future_recon, output_b.future_recon)


def test_future_state_tokenizer_checkpoint_roundtrip() -> None:
    config = _config(zero_init_decoder=False)
    tokenizer = FutureStateTokenizer(config).eval()
    future, token_mask, horizon_ids, camera_ids = _inputs()
    with torch.no_grad():
        output_before = tokenizer(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    buffer = io.BytesIO()
    torch.save({"config": tokenizer.config_dict(), "state_dict": tokenizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    restored = FutureStateTokenizer(DeltaTokenizerConfig(**payload["config"])).eval()
    restored.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        output_after = restored(future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert torch.allclose(output_before.frame_tokens, output_after.frame_tokens)
    assert torch.allclose(output_before.future_recon, output_after.future_recon)


def test_future_state_tokenizer_tiny_overfit_smoke() -> None:
    torch.manual_seed(9)
    config = DeltaTokenizerConfig(
        teacher_dim=8,
        hidden_dim=16,
        delta_dim=4,
        num_delta_tokens=2,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        max_tokens=5,
        max_horizon=4,
        relation_max_tokens=5,
        zero_init_decoder=False,
    )
    tokenizer = FutureStateTokenizer(config)
    future = torch.randn(4, 5, 8)
    horizon_ids = torch.ones(4, dtype=torch.long)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-2, weight_decay=0.0)
    losses = []

    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss_output = future_state_tokenizer_loss(
            tokenizer,
            future,
            horizon_ids=horizon_ids,
            cosine_weight=0.0,
            relation_weight=0.0,
        )
        loss_output.loss.backward()
        optimizer.step()
        losses.append(float(loss_output.loss.detach()))

    assert losses[-1] < losses[0] * 0.75
