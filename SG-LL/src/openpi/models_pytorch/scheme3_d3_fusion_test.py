import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3D3ActionDeltaAdapter
from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme3D3DeltaSuffixFusion


def _fusion_inputs():
    torch.manual_seed(0)
    suffix_out = torch.randn(2, 4, 16)
    delta_tokens = torch.randn(2, 3, 8)
    return suffix_out, delta_tokens


def _adapter() -> Scheme3D3ActionDeltaAdapter:
    return Scheme3D3ActionDeltaAdapter(
        current_dim=12,
        suffix_dim=16,
        delta_dim=8,
        num_delta_tokens=3,
        action_horizon=4,
        action_dim=5,
        state_dim=7,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        max_horizon=8,
        horizon_id=2,
        dropout=0.0,
    )


def _adapter_inputs():
    torch.manual_seed(1)
    batch_size = 2
    suffix_out = torch.randn(batch_size, 4, 16)
    current_tokens = torch.randn(batch_size, 5, 12)
    estimated_actions = torch.randn(batch_size, 4, 5)
    state = torch.randn(batch_size, 7)
    token_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    return suffix_out, current_tokens, estimated_actions, state, token_mask


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in module.parameters()
    )


def test_delta_suffix_fusion_shape_and_zero_gate_identity() -> None:
    fusion = Scheme3D3DeltaSuffixFusion(suffix_dim=16, delta_dim=8, num_heads=4, dropout=0.0)
    suffix_out, delta_tokens = _fusion_inputs()

    refined = fusion(suffix_out, delta_tokens)

    assert refined.shape == suffix_out.shape
    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)
    assert float(fusion.gate.detach().abs().max()) == 0.0


def test_delta_suffix_fusion_rejects_bad_shapes() -> None:
    fusion = Scheme3D3DeltaSuffixFusion(suffix_dim=16, delta_dim=8, num_heads=4, dropout=0.0)
    suffix_out, delta_tokens = _fusion_inputs()

    with pytest.raises(ValueError, match="suffix_out"):
        fusion(suffix_out[:, :, :-1], delta_tokens)

    with pytest.raises(ValueError, match="delta_tokens"):
        fusion(suffix_out, delta_tokens[:, :, :-1])


def test_action_delta_adapter_shapes_and_zero_gate_identity() -> None:
    adapter = _adapter()
    suffix_out, current_tokens, estimated_actions, state, token_mask = _adapter_inputs()

    refined, pred_delta = adapter(
        suffix_out,
        current_tokens,
        estimated_actions,
        state=state,
        token_mask=token_mask,
    )

    assert refined.shape == suffix_out.shape
    assert pred_delta.shape == (2, 3, 8)
    assert torch.allclose(refined, suffix_out.float(), atol=0.0, rtol=0.0)
    assert float(adapter.gate_abs_mean().detach()) == 0.0
    assert float(adapter.gate_max_abs().detach()) == 0.0


def test_action_delta_adapter_zero_gate_gradient_reaches_gate() -> None:
    adapter = _adapter()
    suffix_out, current_tokens, estimated_actions, state, token_mask = _adapter_inputs()

    refined, _pred_delta = adapter(
        suffix_out,
        current_tokens,
        estimated_actions,
        state=state,
        token_mask=token_mask,
    )
    loss = refined.square().mean()
    loss.backward()

    gate_grad = adapter.suffix_fusion.gate.grad
    assert gate_grad is not None
    assert torch.isfinite(gate_grad).all()
    assert gate_grad.abs().sum() > 0


def test_action_delta_adapter_open_gate_gradient_reaches_predictor() -> None:
    adapter = _adapter()
    adapter.suffix_fusion.gate.data.fill_(0.1)
    suffix_out, current_tokens, estimated_actions, state, token_mask = _adapter_inputs()

    refined, pred_delta = adapter(
        suffix_out,
        current_tokens,
        estimated_actions,
        state=state,
        token_mask=token_mask,
    )
    loss = refined.square().mean() + 0.1 * pred_delta.square().mean()
    loss.backward()

    assert _has_nonzero_grad(adapter.predictor)
    assert _has_nonzero_grad(adapter.suffix_fusion)


def test_action_delta_adapter_deterministic_eval() -> None:
    adapter = _adapter().eval()
    suffix_out, current_tokens, estimated_actions, state, token_mask = _adapter_inputs()
    adapter.suffix_fusion.gate.data.fill_(0.1)

    with torch.no_grad():
        refined_a, delta_a = adapter(
            suffix_out,
            current_tokens,
            estimated_actions,
            state=state,
            token_mask=token_mask,
        )
        refined_b, delta_b = adapter(
            suffix_out,
            current_tokens,
            estimated_actions,
            state=state,
            token_mask=token_mask,
        )

    assert torch.allclose(refined_a, refined_b)
    assert torch.allclose(delta_a, delta_b)


def test_action_delta_adapter_tiny_overfit_smoke() -> None:
    torch.manual_seed(2)
    adapter = _adapter()
    adapter.suffix_fusion.gate.data.fill_(0.2)
    suffix_out, current_tokens, estimated_actions, state, token_mask = _adapter_inputs()
    target_proj = torch.randn(
        estimated_actions.numel() // estimated_actions.shape[0],
        suffix_out.numel() // suffix_out.shape[0],
    )
    target_update = torch.tanh(estimated_actions.reshape(estimated_actions.shape[0], -1) @ target_proj * 0.1)
    target_suffix = suffix_out + target_update.reshape_as(suffix_out)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-2, weight_decay=0.0)

    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        refined, _pred_delta = adapter(
            suffix_out,
            current_tokens,
            estimated_actions,
            state=state,
            token_mask=token_mask,
        )
        loss = F.mse_loss(refined, target_suffix)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5
