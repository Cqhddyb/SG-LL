"""Versioned request keys for optional common-random-number evaluation."""

import hashlib
import json

NOISE_SEED_KEY = "_policy_noise_seed"
NOISE_PROTOCOL = "episode_request_sha256_v1"


def episode_request_seed(
    *, suite: str, task_id: int, episode_idx: int, env_seed: int, policy_seed: int, request_index: int
) -> int:
    """Return a stable 63-bit seed, independent of mode, shard and request order.

    Request index is the action-chunk request within one episode, not an
    environment step. Common noise does not force diverged states to agree.
    """
    values = (task_id, episode_idx, env_seed, policy_seed, request_index)
    if not suite or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Request noise requires a suite and nonnegative integer episode/request keys")
    payload = json.dumps([NOISE_PROTOCOL, suite, *values], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
