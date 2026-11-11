"""Validation for selecting a contiguous window of official LIBERO initial states."""

from __future__ import annotations


def resolve_initial_state_indices(
    *,
    trial_start: int,
    num_trials_per_task: int,
    num_initial_states: int,
) -> range:
    """Return the official initial-state indices for one task evaluation.

    The default caller values (``trial_start=0`` and ``num_trials_per_task=50``)
    reproduce the original full standard-LIBERO loop. A nonzero ``trial_start``
    is explicit because result ``episode_idx`` must identify the underlying
    official initial state, not merely the position inside a sliced run.
    """
    if trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {trial_start}")
    if num_trials_per_task <= 0:
        raise ValueError(f"num_trials_per_task must be positive, got {num_trials_per_task}")
    if num_initial_states <= 0:
        raise ValueError(f"num_initial_states must be positive, got {num_initial_states}")

    trial_stop = trial_start + num_trials_per_task
    if trial_stop > num_initial_states:
        raise ValueError(
            "Requested initial-state window "
            f"[{trial_start}, {trial_stop}) exceeds {num_initial_states} available official initial states."
        )
    return range(trial_start, trial_stop)
