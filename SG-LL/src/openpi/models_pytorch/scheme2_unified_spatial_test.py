import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme2UnifiedSpatialBottleneckAdapter


def _adapter() -> Scheme2UnifiedSpatialBottleneckAdapter:
    return Scheme2UnifiedSpatialBottleneckAdapter(
        visual_dim=12,
        spatial_dim=10,
        hidden_dim=24,
        max_aux_tokens=4,
        dropout=0.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    vision_hidden = torch.randn(2, 6, 12)
    vision_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )
    return vision_hidden, vision_mask


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_unified_spatial_adapter_shapes_and_zero_gate_identity() -> None:
    adapter = _adapter()
    vision_hidden, vision_mask = _inputs()

    spatial_tokens, aux_tokens, aux_mask = adapter(vision_hidden, vision_mask, select_aux=False)

    assert spatial_tokens.shape == (2, 6, 10)
    assert aux_tokens.shape == vision_hidden.shape
    assert aux_mask is vision_mask
    assert torch.allclose(aux_tokens, torch.zeros_like(aux_tokens), atol=0.0, rtol=0.0)
    assert float(adapter.gate.detach().abs().max()) == 0.0


def test_unified_spatial_adapter_subsamples_aux_tokens_explicitly() -> None:
    adapter = _adapter()
    vision_hidden, vision_mask = _inputs()

    spatial_tokens, aux_tokens, aux_mask = adapter(vision_hidden, vision_mask, select_aux=True)

    assert spatial_tokens.shape == (2, 6, 10)
    assert aux_tokens.shape == (2, 4, 12)
    assert aux_mask is not None
    assert aux_mask.shape == (2, 4)


def test_unified_spatial_adapter_rejects_bad_shapes() -> None:
    adapter = _adapter()
    vision_hidden, vision_mask = _inputs()

    with pytest.raises(ValueError, match="vision_hidden"):
        adapter(vision_hidden[:, :, :-1], vision_mask)

    with pytest.raises(ValueError, match="vision_mask"):
        adapter(vision_hidden, vision_mask[:, :-1])


def test_unified_spatial_adapter_zero_gate_gradient_reaches_gate() -> None:
    adapter = _adapter()
    vision_hidden, vision_mask = _inputs()
    vision_hidden = vision_hidden.requires_grad_()

    _spatial_tokens, aux_tokens, _aux_mask = adapter(vision_hidden, vision_mask, select_aux=False)
    prefix_after_residual = vision_hidden + aux_tokens
    loss = prefix_after_residual.square().mean()
    loss.backward()

    assert adapter.gate.grad is not None
    assert torch.isfinite(adapter.gate.grad).all()
    assert adapter.gate.grad.abs().sum() > 0
    assert vision_hidden.grad is not None
    assert torch.isfinite(vision_hidden.grad).all()


def test_unified_spatial_adapter_open_gate_gradient_reaches_bottleneck() -> None:
    adapter = _adapter()
    adapter.gate.data.fill_(0.1)
    vision_hidden, vision_mask = _inputs()

    spatial_tokens, aux_tokens, _aux_mask = adapter(vision_hidden, vision_mask, select_aux=False)
    loss = aux_tokens.square().mean() + 0.1 * spatial_tokens.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.to_spatial)
    assert _has_nonzero_grad(adapter.to_action_residual)


def test_unified_spatial_adapter_deterministic_eval() -> None:
    adapter = _adapter().eval()
    adapter.gate.data.fill_(0.1)
    vision_hidden, vision_mask = _inputs()

    with torch.no_grad():
        spatial_a, aux_a, mask_a = adapter(vision_hidden, vision_mask, select_aux=False)
        spatial_b, aux_b, mask_b = adapter(vision_hidden, vision_mask, select_aux=False)

    assert mask_a is vision_mask
    assert mask_b is vision_mask
    assert torch.allclose(spatial_a, spatial_b)
    assert torch.allclose(aux_a, aux_b)


def test_unified_spatial_adapter_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    adapter = _adapter()
    adapter.gate.data.fill_(0.2)
    vision_hidden, vision_mask = _inputs()
    target_proj = torch.randn(vision_hidden.shape[-1], vision_hidden.shape[-1])
    target_aux = torch.tanh(vision_hidden @ target_proj * 0.1)
    target_aux = target_aux * vision_mask.to(dtype=target_aux.dtype).unsqueeze(-1)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        _spatial_tokens, aux_tokens, _aux_mask = adapter(vision_hidden, vision_mask, select_aux=False)
        loss = F.mse_loss(aux_tokens, target_aux)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5
