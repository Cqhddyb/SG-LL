"""Numerical/provenance checks for the bounded diagnostic, without model load."""

import copy

import numpy as np
import pytest
import torch

from scripts.diagnose_scheme4_utility import action_chunk_for_forward
from scripts.diagnose_scheme4_utility import half_fusion_velocity
from scripts.diagnose_scheme4_utility import make_samples
from scripts.diagnose_scheme4_utility import summarize


def metadata():
    tasks = [{"task_index": i, "task": f"task{i}"} for i in range(2)]
    episodes = [{"episode_index": i, "tasks": [f"task{i % 2}"], "length": 40 + i} for i in range(40)]
    return episodes, tasks


def test_trajectory_partitions_and_midpoints():
    episodes, tasks = metadata()
    samples, validation, frames = make_samples(episodes, tasks, split_seed=3, action_horizon=10)
    assert (samples, validation, frames) == make_samples(episodes, tasks, split_seed=3, action_horizon=10)
    assert frames == sum(ep["length"] for ep in episodes)
    assert len(validation) == 4
    assert len(samples) == 4
    for row in samples:
        assert (row["episode_index"] in validation) == (row["split"] == "prospective_validation")
        assert row["global_index"] == sum(ep["length"] for ep in episodes[: row["episode_index"]]) + row["frame_index"]
        assert row["frame_index"] + 10 <= row["episode_length"]
    assert len({r["episode_index"] for r in samples}) == 4


def test_reject_unordered_or_ambiguous_metadata():
    episodes, tasks = metadata()
    bad = copy.deepcopy(episodes)
    bad[0]["episode_index"] = 3
    with pytest.raises(ValueError, match="ordered"):
        make_samples(bad, tasks, split_seed=3, action_horizon=10)
    episodes[0]["tasks"].append("task1")
    with pytest.raises(ValueError, match="single-task"):
        make_samples(episodes, tasks, split_seed=3, action_horizon=10)


def test_reject_no_unpadded_chunk():
    episodes, tasks = metadata()
    for ep in episodes:
        ep["length"] = 5
    with pytest.raises(ValueError, match="unpadded"):
        make_samples(episodes, tasks, split_seed=3, action_horizon=10)


def test_summary_known_utility_and_constants():
    rows = []
    # At truth=0, base=2, full=-1, true half loss=.25; weighted losses=2.5.
    for split in ("development", "prospective_validation"):
        for draw, (target, prob, benefit) in enumerate(((0.75, 0.8, 3), (0.25, 0.2, -3))):
            base, full = (4, 1) if benefit > 0 else (1, 4)
            rows.append(
                {
                    "split": split,
                    "episode_index": 1,
                    "draw": draw,
                    "target": target,
                    "probability": prob,
                    "benefit": benefit,
                    "loss_none": base,
                    "loss_fixed": full,
                    "loss_half": 0.25,
                    "loss_soft": 0.16,
                    "loss_hard": full if prob >= 0.5 else base,
                    "loss_oracle_hard": 1,
                }
            )
    summary = summarize(rows)
    assert summary["development_mean_target"] == 0.5
    for split in summary["splits"].values():
        assert split["two_draw_sign_agreement"] == 0
        assert split["calibration"]["router"]["sign_accuracy"] == 1
        assert split["calibration"]["router"]["brier"] == pytest.approx(0.0025)
        assert split["calibration"]["half"]["bce"] == pytest.approx(np.log(2))
        assert split["action_mse"]["half"] == 0.25
        assert split["action_mse"]["hard"] == 1
    # Prospective target changes must not leak into the development constant.
    for row in rows:
        if row["split"] == "prospective_validation":
            row["target"] = 0.99
    assert summarize(rows)["development_mean_target"] == 0.5


def test_summary_requires_development():
    with pytest.raises(ValueError, match="Development"):
        summarize([])


def test_normalized_float64_actions_match_training_float32_cast():
    actions = torch.arange(32, dtype=torch.float64).reshape(1, 2, 16) / 3
    result = action_chunk_for_forward(actions, horizon=2, action_dim=16)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, actions.float(), rtol=0, atol=0)
    # The original failure happened at a float32 action projection.
    assert torch.isfinite(torch.nn.Linear(16, 8)(result)).all()
    with pytest.raises(ValueError, match="shape"):
        action_chunk_for_forward(actions, horizon=3, action_dim=16)


def test_half_control_uses_actual_projection_with_rounding():
    class Adapter:
        @staticmethod
        def suffix_fusion(suffix, delta, *, residual_scale):
            return suffix + residual_scale * delta

    # Rounding represents a finite-precision projection, not an experiment.
    def project(x):
        return (x * 10).round() / 10

    base, delta = torch.tensor([0.14]), torch.tensor([0.08])
    actual = half_fusion_velocity(Adapter(), project, base, delta)
    torch.testing.assert_close(actual, torch.tensor([0.2]))
    interpolated_outputs = (project(base) + project(base + delta)) / 2
    torch.testing.assert_close(interpolated_outputs, torch.tensor([0.15]))
    assert not torch.equal(actual, interpolated_outputs)
