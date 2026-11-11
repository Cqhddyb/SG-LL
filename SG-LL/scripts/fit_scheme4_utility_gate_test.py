"""Isolation, numerical utility, gradient and fit checks; no VLA weights."""

import copy

import pytest
import torch

from openpi.models_pytorch.scheme4_adaptive_lookahead import Scheme4UtilityRouter
from scripts.diagnose_scheme4_utility import make_samples
from scripts.fit_scheme4_utility_gate import constant_gate
from scripts.fit_scheme4_utility_gate import fit_router
from scripts.fit_scheme4_utility_gate import paired_interval
from scripts.fit_scheme4_utility_gate import pooled_inputs
from scripts.fit_scheme4_utility_gate import router_forward
from scripts.fit_scheme4_utility_gate import select_samples
from scripts.fit_scheme4_utility_gate import utility_coefficients
from scripts.fit_scheme4_utility_gate import utility_risk
from scripts.fit_scheme4_utility_gate import verify_isolation


def fixture_metadata():
    tasks = [{"task": f"task{i}", "task_index": i} for i in range(2)]
    episodes = [{"episode_index": i, "tasks": [f"task{i % 2}"], "length": 60 + i} for i in range(80)]
    samples, reserved, frames = make_samples(episodes, tasks, split_seed=20261110, action_horizon=10)
    return (
        episodes,
        tasks,
        {"samples": samples, "validation_episode_ids": reserved, "total_frames": frames, "split_seed": 20261110},
    )


def test_sample_manifest_reuses_split_excludes_whole_trajectories_and_old_validation():
    episodes, tasks, prior = fixture_metadata()
    rows = select_samples(episodes, tasks, prior, action_horizon=10)
    assert rows == select_samples(episodes, tasks, prior, action_horizon=10)
    assert sum(r["split"] == "train" for r in rows) == 16
    assert sum(r["split"] == "validation" for r in rows) == 4
    old_val = {r["episode_index"] for r in prior["samples"] if r["split"] == "prospective_validation"}
    for row in rows:
        assert (row["episode_index"] in prior["validation_episode_ids"]) == (row["split"] == "validation")
        assert row["frame_index"] + 10 <= row["episode_length"]
        assert row["global_index"] == sum(ep["length"] for ep in episodes[: row["episode_index"]]) + row["frame_index"]
        if row["split"] == "validation":
            assert row["episode_index"] not in old_val
    bad = copy.deepcopy(rows)
    bad[0]["episode_index"] = prior["validation_episode_ids"][0]
    with pytest.raises(ValueError, match="isolation"):
        verify_isolation(bad, prior["validation_episode_ids"])
    changed = {**prior, "total_frames": prior["total_frames"] + 1}
    with pytest.raises(ValueError, match="split changed"):
        select_samples(episodes, tasks, changed, action_horizon=10)
    smoke = select_samples(episodes, tasks, prior, action_horizon=10, smoke=True)
    assert len(smoke) == 2
    assert all(row in rows for row in smoke)


def tiny_router_and_features():
    torch.manual_seed(42)
    router = Scheme4UtilityRouter(spatial_dim=4, language_dim=3, state_dim=2, action_intent_dim=4, hidden_dim=8)
    features = {
        "spatial_tokens": torch.randn(8, 1, 4),
        "language_features": torch.randn(8, 3),
        "state": torch.randn(8, 2),
        "action_intent": torch.randn(8, 4),
    }
    return router, features


def test_pooling_replays_exact_router_and_detaches_all_inputs():
    router, features = tiny_router_and_features()
    spatial = features.pop("spatial_tokens").expand(-1, 5, -1).clone().requires_grad_(requires_grad=True)
    for x in features.values():
        x.requires_grad_(requires_grad=True)
    mask = torch.tensor([[True, False, True, False, True]] * 7 + [[False] * 5])
    expected = router(spatial, token_mask=mask, **features)
    pooled = pooled_inputs((spatial,), {**features, "token_mask": mask})
    torch.testing.assert_close(router_forward(router, pooled), expected, rtol=0, atol=0)
    assert not any(x.requires_grad for x in pooled.values())
    assert torch.count_nonzero(pooled["spatial_tokens"][-1]) == 0


def test_utility_quadratic_matches_real_affine_mse_and_gradients():
    base = torch.tensor([[[2.0]], [[1.0]]], requires_grad=True)
    full = torch.tensor([[[-1.0]], [[3.0]]], requires_grad=True)
    truth = torch.zeros_like(base)
    q, c = utility_coefficients(truth, base, full)
    gate = torch.tensor([0.5, 0.25], requires_grad=True)
    loss = utility_risk(gate, q, c) + base.detach().square().mean((1, 2))
    actual = (truth - (base.detach() + gate[:, None, None] * (full.detach() - base.detach()))).square().mean((1, 2))
    torch.testing.assert_close(loss, actual)
    loss.sum().backward()
    torch.testing.assert_close(gate.grad, torch.tensor([-3.0, 6.0]))
    assert base.grad is None
    assert full.grad is None
    assert not q.requires_grad
    assert not c.requires_grad
    assert constant_gate(q, c) == pytest.approx(4 / 13)
    assert constant_gate(torch.zeros(2), torch.zeros(2)) == 0
    assert constant_gate(torch.ones(2), torch.tensor([3.0, 2.0])) == 1
    with pytest.raises(ValueError, match="shape"):
        utility_coefficients(truth, base[:, 0], full)
    with pytest.raises(ValueError, match="shape"):
        utility_risk(torch.ones(2, 1), q, c)
    with pytest.raises(ValueError, match="negative curvature"):
        utility_risk(torch.ones(2), -q, c)


def test_tiny_overfit_is_deterministic_and_reaches_all_encoders():
    router, features = tiny_router_and_features()
    target = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    q = torch.ones(8)
    initial_error = (router_forward(router, features) - target).square().mean().item()
    settings = {"seed": 42, "steps": 150, "batch_size": 8, "lr": 0.01, "weight_decay": 0.0, "warmup_steps": 1}
    trace = fit_router(router, features, q, target, settings=settings, normalization=1.0)
    predicted = router_forward(router, features)
    assert (predicted - target).square().mean().item() < initial_error / 10
    assert all(p.grad is not None for p in router.parameters())
    assert not any(p.requires_grad for p in router.parameters())
    other, _ = tiny_router_and_features()
    assert trace == fit_router(other, features, q, target, settings=settings, normalization=1.0)
    torch.testing.assert_close(router_forward(other, features), predicted, rtol=0, atol=0)
    with pytest.raises(ValueError, match="detached"):
        fit_router(
            other,
            {**features, "state": features["state"].requires_grad_(requires_grad=True)},
            q,
            target,
            settings=settings,
            normalization=1.0,
        )


def test_bootstrap_pairs_draws_within_episode_and_is_seeded():
    rows = [
        {"episode_index": episode, "loss_fixed": fixed, "loss_soft": 1.0}
        for episode, fixed in ((0, 1.5), (0, 2.5), (1, 3), (1, 3))
    ]
    report = paired_interval(rows, "fixed", seed=42)
    assert report == paired_interval(rows, "fixed", seed=42)
    assert report["episodes"] == 2
    assert report["mean_control_minus_soft"] == 1.5
    assert report["ci95"] == [1.0, 2.0]
