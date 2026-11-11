"""Request noise contracts through real Policy transforms and sampling."""

from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest
import torch

from openpi.policies.policy import Policy
from openpi.shared.request_noise import NOISE_PROTOCOL
from openpi.shared.request_noise import NOISE_SEED_KEY
from openpi.shared.request_noise import episode_request_seed


class _NoiseModel(torch.nn.Module):
    config = SimpleNamespace(action_horizon=50, action_dim=32)

    def sample_actions(self, device, observation, noise=None):
        assert observation.state.shape == (1, 32)
        if noise is None:
            return torch.randn((1, 50, 32), device=device)
        assert noise.shape == (1, 50, 32)
        assert noise.dtype == torch.float32
        return noise.clone()


def _policy(**kwargs):
    def check_transform(obs):
        assert NOISE_SEED_KEY not in obs
        return obs

    return Policy(_NoiseModel(), is_pytorch=True, transforms=[check_transform], **kwargs)


def _obs(seed=None):
    obs = {"image": {}, "image_mask": {}, "state": np.zeros(32, dtype=np.float32)}
    if seed is not None:
        obs[NOISE_SEED_KEY] = seed
    return obs


def _seed(**overrides):
    keys = {
        "suite": "libero_spatial",
        "task_id": 0,
        "episode_idx": 0,
        "env_seed": 7,
        "policy_seed": 7,
        "request_index": 0,
    }
    return episode_request_seed(**(keys | overrides))


def test_request_seed_is_stable_across_serialization_and_distinguishes_episode_keys():
    seed = _seed()
    assert 0 <= seed < 2**63
    assert seed == _seed()
    assert msgpack_numpy.unpackb(msgpack_numpy.Packer().pack({NOISE_SEED_KEY: seed}))[NOISE_SEED_KEY] == seed
    alternatives = [
        _seed(suite="libero_10"),
        _seed(task_id=1),
        _seed(episode_idx=1),
        _seed(env_seed=8),
        _seed(policy_seed=8),
        _seed(request_index=1),
    ]
    assert len({seed, *alternatives}) == 7


def test_request_noise_is_order_independent_and_does_not_advance_global_rng():
    policy = _policy()
    request = _obs(_seed())
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    a = policy.infer(request)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert request[NOISE_SEED_KEY] == _seed()  # Input was not mutated.
    policy.infer(_obs(_seed(task_id=3)))  # Another client/request.
    torch.randn(10)  # Unrelated global sampling.
    b = policy.infer(request)
    assert np.array_equal(a["actions"], b["actions"])
    assert a["action_noise"] == b["action_noise"]
    assert a["action_noise"]["protocol"] == NOISE_PROTOCOL
    assert len(a["action_noise"]["sha256"]) == 64
    assert np.isfinite(a["actions"]).all()
    assert not np.array_equal(a["actions"], policy.infer(_obs(_seed(request_index=1)))["actions"])
    # Explicit legacy noise and seeded noise enter exactly the same sampler.
    c = policy.infer(_obs(), noise=a["actions"])
    assert np.array_equal(c["actions"], a["actions"])
    assert "action_noise" not in c


def test_legacy_sampling_keeps_original_global_rng_behavior():
    policy = _policy()
    torch.manual_seed(17)
    expected = torch.randn((1, 50, 32)).numpy()[0]
    torch.manual_seed(17)
    actual = policy.infer(_obs())
    assert np.array_equal(actual["actions"], expected)
    assert "action_noise" not in actual


@pytest.mark.parametrize("seed", [-1, 2**63, True, "7", 0.2, None])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="Request noise seed must"):
        _policy().infer({**_obs(), NOISE_SEED_KEY: seed})


def test_explicit_noise_cannot_silently_override_request_seed():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _policy().infer(_obs(7), noise=np.zeros((50, 32), dtype=np.float32))
    policy = _policy(sample_kwargs={"noise": torch.zeros((1, 50, 32))})
    with pytest.raises(ValueError, match="mutually exclusive"):
        policy.infer(_obs(7))


def test_jax_request_seed_fails_explicitly(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(policy, "_is_pytorch_model", False)
    with pytest.raises(ValueError, match="PyTorch policy"):
        policy.infer(_obs(7))


@pytest.mark.parametrize("override", [{"task_id": -1}, {"request_index": True}, {"policy_seed": -1}, {"suite": ""}])
def test_invalid_episode_keys_rejected(override):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _seed(**override)
