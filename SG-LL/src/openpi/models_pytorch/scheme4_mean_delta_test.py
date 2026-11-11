import pytest
import torch

from openpi.models_pytorch.scheme4_mean_delta import DeltaTokenMeanAccumulator
from openpi.models_pytorch.scheme4_mean_delta import deterministic_subset_indices


def test_delta_token_mean_accumulator_shape_detach_and_statistics() -> None:
    source = torch.arange(48, dtype=torch.float32, requires_grad=True).view(2, 3, 8)
    accumulator = DeltaTokenMeanAccumulator((3, 8))
    accumulator.update(source)
    mean, stats = accumulator.finalize()

    assert mean.shape == (3, 8)
    assert mean.device.type == "cpu"
    assert not mean.requires_grad
    assert torch.allclose(mean, source.detach().mean(dim=0))
    assert stats["sample_count"] == 2
    assert stats["token_shape"] == [3, 8]


def test_delta_token_mean_accumulator_is_batch_partition_deterministic() -> None:
    generator = torch.Generator().manual_seed(9)
    tokens = torch.randn(6, 3, 8, generator=generator)
    all_at_once = DeltaTokenMeanAccumulator((3, 8))
    split = DeltaTokenMeanAccumulator((3, 8))
    all_at_once.update(tokens)
    split.update(tokens[:2])
    split.update(tokens[2:])

    mean_a, stats_a = all_at_once.finalize()
    mean_b, stats_b = split.finalize()
    assert torch.equal(mean_a, mean_b)
    assert stats_a == stats_b


def test_delta_token_mean_accumulator_rejects_invalid_inputs() -> None:
    accumulator = DeltaTokenMeanAccumulator((3, 8))
    with pytest.raises(ValueError, match="shape"):
        accumulator.update(torch.zeros(3, 8))
    with pytest.raises(ValueError, match="expected"):
        accumulator.update(torch.zeros(2, 4, 8))
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update(torch.full((2, 3, 8), float("nan")))
    with pytest.raises(ValueError, match="empty"):
        accumulator.finalize()


def test_deterministic_subset_indices() -> None:
    indices_a = deterministic_subset_indices(100, 16, seed=5)
    indices_b = deterministic_subset_indices(100, 16, seed=5)
    indices_c = deterministic_subset_indices(100, 16, seed=6)
    assert indices_a == indices_b
    assert indices_a != indices_c
    assert indices_a == sorted(set(indices_a))
    assert len(indices_a) == 16
