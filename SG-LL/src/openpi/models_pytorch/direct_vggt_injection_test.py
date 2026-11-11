import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_proposal_variants_pytorch import DirectVGGTInjectionAdapter


def _adapter() -> DirectVGGTInjectionAdapter:
    return DirectVGGTInjectionAdapter(
        teacher_dim=10,
        visual_dim=12,
        hidden_dim=24,
        max_aux_tokens=4,
        dropout=0.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    teacher_tokens = torch.randn(2, 6, 10)
    token_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )
    return teacher_tokens, token_mask


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_direct_vggt_adapter_shapes_and_zero_gate_identity() -> None:
    adapter = _adapter()
    teacher_tokens, token_mask = _inputs()

    aux_tokens, aux_mask = adapter(teacher_tokens, token_mask, select_aux=False)

    assert aux_tokens.shape == (2, 6, 12)
    assert aux_mask is token_mask
    assert torch.allclose(aux_tokens, torch.zeros_like(aux_tokens), atol=0.0, rtol=0.0)
    assert float(adapter.gate.detach().abs().max()) == 0.0


def test_direct_vggt_adapter_subsamples_aux_tokens_explicitly() -> None:
    adapter = _adapter()
    teacher_tokens, token_mask = _inputs()

    aux_tokens, aux_mask = adapter(teacher_tokens, token_mask, select_aux=True)

    assert aux_tokens.shape == (2, 4, 12)
    assert aux_mask is not None
    assert aux_mask.shape == (2, 4)


def test_direct_vggt_adapter_rejects_bad_shapes() -> None:
    adapter = _adapter()
    teacher_tokens, token_mask = _inputs()

    with pytest.raises(ValueError, match="teacher_tokens"):
        adapter(teacher_tokens[:, :, :-1], token_mask)

    with pytest.raises(ValueError, match="token_mask"):
        adapter(teacher_tokens, token_mask[:, :-1])


def test_direct_vggt_adapter_zero_gate_gradient_reaches_gate() -> None:
    adapter = _adapter()
    teacher_tokens, token_mask = _inputs()
    prefix_tokens = torch.randn(teacher_tokens.shape[0], teacher_tokens.shape[1], 12)

    aux_tokens, _aux_mask = adapter(teacher_tokens, token_mask, select_aux=False)
    loss = (prefix_tokens + aux_tokens).square().mean()
    loss.backward()

    assert adapter.gate.grad is not None
    assert torch.isfinite(adapter.gate.grad).all()
    assert adapter.gate.grad.abs().sum() > 0


def test_direct_vggt_adapter_open_gate_gradient_reaches_projection() -> None:
    adapter = _adapter()
    adapter.gate.data.fill_(0.1)
    teacher_tokens, token_mask = _inputs()

    aux_tokens, _aux_mask = adapter(teacher_tokens, token_mask, select_aux=False)
    loss = aux_tokens.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.to_action_residual)


def test_direct_vggt_adapter_deterministic_eval() -> None:
    adapter = _adapter().eval()
    adapter.gate.data.fill_(0.1)
    teacher_tokens, token_mask = _inputs()

    with torch.no_grad():
        aux_a, mask_a = adapter(teacher_tokens, token_mask, select_aux=False)
        aux_b, mask_b = adapter(teacher_tokens, token_mask, select_aux=False)

    assert mask_a is token_mask
    assert mask_b is token_mask
    assert torch.allclose(aux_a, aux_b)


def test_direct_vggt_adapter_tiny_overfit_smoke() -> None:
    torch.manual_seed(1)
    adapter = _adapter()
    adapter.gate.data.fill_(0.2)
    teacher_tokens, token_mask = _inputs()
    target_proj = torch.randn(teacher_tokens.shape[-1], 12)
    target_aux = torch.tanh(teacher_tokens @ target_proj * 0.1)
    target_aux = target_aux * token_mask.to(dtype=target_aux.dtype).unsqueeze(-1)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        aux_tokens, _aux_mask = adapter(teacher_tokens, token_mask, select_aux=False)
        loss = F.mse_loss(aux_tokens, target_aux)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5
