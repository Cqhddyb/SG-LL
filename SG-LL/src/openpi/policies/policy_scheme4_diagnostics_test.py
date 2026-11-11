import pytest
import torch

from openpi.policies.policy import _consume_pytorch_inference_diagnostics


class _FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def take_scheme4_inference_diagnostics(self):
        self.calls += 1
        payload = self.payload
        self.payload = None
        return payload


def test_consumer_preserves_disabled_response_absence() -> None:
    model = _FakeModel(None)
    assert _consume_pytorch_inference_diagnostics(model) is None
    assert model.calls == 1


def test_consumer_serializes_batch_one_scalars_and_flow_steps() -> None:
    model = _FakeModel(
        {
            "fusion_scale": torch.tensor([1.0]),
            "hidden_residual_l2_mean": torch.tensor([[0.1, 0.2, 0.3]]),
        }
    )
    assert _consume_pytorch_inference_diagnostics(model) == {
        "fusion_scale": 1.0,
        "hidden_residual_l2_mean": pytest.approx([0.1, 0.2, 0.3]),
    }
    assert _consume_pytorch_inference_diagnostics(model) is None


def test_consumer_rejects_implicit_batch_selection() -> None:
    model = _FakeModel({"fusion_scale": torch.ones(2)})
    with pytest.raises(RuntimeError, match=r"must be \[1\]"):
        _consume_pytorch_inference_diagnostics(model)
