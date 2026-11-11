from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("initial_state_window.py")
_SPEC = importlib.util.spec_from_file_location("initial_state_window", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_default_window_preserves_full_fifty_state_protocol() -> None:
    indices = _MODULE.resolve_initial_state_indices(
        trial_start=0,
        num_trials_per_task=50,
        num_initial_states=50,
    )

    assert list(indices) == list(range(50))


def test_held_out_window_uses_underlying_official_indices() -> None:
    indices = _MODULE.resolve_initial_state_indices(
        trial_start=25,
        num_trials_per_task=25,
        num_initial_states=50,
    )

    assert list(indices) == list(range(25, 50))


@pytest.mark.parametrize(
    ("trial_start", "num_trials_per_task", "num_initial_states", "message"),
    [
        (-1, 25, 50, "non-negative"),
        (0, 0, 50, "positive"),
        (26, 25, 50, "exceeds"),
    ],
)
def test_invalid_windows_are_rejected(
    trial_start: int,
    num_trials_per_task: int,
    num_initial_states: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _MODULE.resolve_initial_state_indices(
            trial_start=trial_start,
            num_trials_per_task=num_trials_per_task,
            num_initial_states=num_initial_states,
        )
