import dataclasses
import io

import pytest
import torch

from openpi.models_pytorch.delta_tokenizer import DeltaTokenizer
from openpi.models_pytorch.delta_tokenizer import DeltaTokenizerConfig
from openpi.models_pytorch.delta_tokenizer import delta_tokenizer_loss


def _small_config(*, zero_init_decoder: bool = True) -> DeltaTokenizerConfig:
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


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    batch_size = 2
    num_tokens = 6
    teacher_dim = 16
    current = torch.randn(batch_size, num_tokens, teacher_dim)
    future = current + 0.1 * torch.randn(batch_size, num_tokens, teacher_dim)
    token_mask = torch.tensor([[True, True, True, True, True, False], [True, True, True, False, False, False]])
    horizon_ids = torch.tensor([1, 3], dtype=torch.long)
    camera_ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    return current, future, token_mask, horizon_ids, camera_ids


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_delta_tokenizer_shapes_and_masks() -> None:
    tokenizer = DeltaTokenizer(_small_config())
    current, future, token_mask, horizon_ids, camera_ids = _inputs()

    output = tokenizer(current, future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert output.delta_tokens.shape == (2, 4, 8)
    assert output.future_recon.shape == current.shape
    assert output.future_recon.dtype == torch.float32


def test_delta_tokenizer_rejects_bad_shapes() -> None:
    tokenizer = DeltaTokenizer(_small_config())
    current, future, token_mask, horizon_ids, camera_ids = _inputs()

    with pytest.raises(ValueError, match="identical shape"):
        tokenizer(current, future[:, :-1], token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    with pytest.raises(ValueError, match="token_mask"):
        tokenizer(current, future, token_mask=token_mask[:, :-1], horizon_ids=horizon_ids, camera_ids=camera_ids)

    with pytest.raises(ValueError, match="horizon_ids"):
        tokenizer(current, future, token_mask=token_mask, horizon_ids=horizon_ids[:, None], camera_ids=camera_ids)

    no_camera_tokenizer = DeltaTokenizer(dataclasses.replace(_small_config(), num_camera_embeddings=0))
    with pytest.raises(ValueError, match="camera_ids"):
        no_camera_tokenizer(current, future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)


def test_zero_delta_identity_with_zero_init_decoder() -> None:
    tokenizer = DeltaTokenizer(_small_config(zero_init_decoder=True))
    current, _future, token_mask, horizon_ids, camera_ids = _inputs()
    zero_delta = torch.zeros(current.shape[0], tokenizer.config.num_delta_tokens, tokenizer.config.delta_dim)

    recon = tokenizer.decode(current, zero_delta, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert torch.equal(recon, current.float())


def test_delta_tokenizer_gradient_flow() -> None:
    tokenizer = DeltaTokenizer(_small_config(zero_init_decoder=False))
    current, future, token_mask, horizon_ids, camera_ids = _inputs()
    current = current.requires_grad_()

    output = tokenizer(current, future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)
    loss = output.future_recon.square().mean() + output.delta_tokens.square().mean()
    loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert _has_nonzero_grad(tokenizer.encoder)
    assert _has_nonzero_grad(tokenizer.decoder)


def test_delta_tokenizer_checkpoint_roundtrip() -> None:
    torch.manual_seed(1)
    config = _small_config(zero_init_decoder=False)
    tokenizer = DeltaTokenizer(config).eval()
    current, future, token_mask, horizon_ids, camera_ids = _inputs()
    with torch.no_grad():
        output_before = tokenizer(
            current, future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids
        )

    buffer = io.BytesIO()
    torch.save({"config": tokenizer.config_dict(), "state_dict": tokenizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)

    restored = DeltaTokenizer(DeltaTokenizerConfig(**payload["config"])).eval()
    restored.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        output_after = restored(current, future, token_mask=token_mask, horizon_ids=horizon_ids, camera_ids=camera_ids)

    assert torch.allclose(output_before.delta_tokens, output_after.delta_tokens)
    assert torch.allclose(output_before.future_recon, output_after.future_recon)


def test_delta_tokenizer_tiny_overfit_smoke() -> None:
    torch.manual_seed(2)
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
    tokenizer = DeltaTokenizer(config)
    current = torch.randn(4, 5, 8)
    future = current + 0.2 * torch.sin(current)
    horizon_ids = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-2)

    losses = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss_output = delta_tokenizer_loss(
            tokenizer,
            current,
            future,
            horizon_ids=horizon_ids,
            cosine_weight=0.0,
            relation_weight=0.0,
            identity_weight=0.0,
        )
        loss_output.loss.backward()
        optimizer.step()
        losses.append(float(loss_output.loss.detach()))

    assert losses[-1] < losses[0] * 0.8
