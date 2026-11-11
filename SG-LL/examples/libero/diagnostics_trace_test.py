import pytest

from openpi.shared.scheme4_diagnostics_trace import FLOW_METRICS
from openpi.shared.scheme4_diagnostics_trace import REQUEST_METRICS
from openpi.shared.scheme4_diagnostics_trace import make_diagnostics_row
from openpi.shared.scheme4_diagnostics_trace import validate_diagnostics_payload


def _payload(flow_steps: int = 3):
    return {
        **{name: [0.1 + index for index in range(flow_steps)] for name in FLOW_METRICS},
        **dict.fromkeys(REQUEST_METRICS, 1.0),
    }


def test_make_diagnostics_row_has_exact_episode_and_request_key() -> None:
    row = make_diagnostics_row(
        run_id="run",
        suite="libero_10",
        task_id=3,
        episode_idx=7,
        seed=7,
        policy_request_index=2,
        env_step=20,
        replan_steps=5,
        diagnostics=_payload(),
    )
    assert row["type"] == "scheme4_policy_request_diagnostics"
    assert (row["suite"], row["task_id"], row["episode_idx"], row["seed"]) == ("libero_10", 3, 7, 7)
    assert row["policy_request_index"] == 2
    assert len(row["diagnostics"]["hidden_residual_l2_mean"]) == 3


def test_validate_diagnostics_payload_rejects_missing_and_misaligned_vectors() -> None:
    missing = _payload()
    missing.pop("fusion_scale")
    with pytest.raises(ValueError, match="keys differ"):
        validate_diagnostics_payload(missing)
    misaligned = _payload()
    misaligned["velocity_delta_l2_mean"] = [0.1]
    with pytest.raises(ValueError, match="same number of steps"):
        validate_diagnostics_payload(misaligned)


def test_adaptive_diagnostics_preserve_legacy_schema_and_routing_values() -> None:
    legacy = _payload()
    assert validate_diagnostics_payload(legacy) == legacy
    adaptive = {**legacy, "lookahead_probability": 0.4, "lookahead_scale": 0.0, "fusion_scale": 0.0}
    assert validate_diagnostics_payload(adaptive) == adaptive
    # The global multiplier is separate from the bounded router gate.
    scaled = {**adaptive, "lookahead_scale": 0.8, "fusion_scale": 1.2}
    assert validate_diagnostics_payload(scaled) == scaled


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["lookahead_probability", "lookahead_scale"])
def test_adaptive_diagnostics_reject_invalid_routing_values(field: str, value: float) -> None:
    payload = {**_payload(), "lookahead_probability": 0.4, "lookahead_scale": 0.4, field: value}
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        validate_diagnostics_payload(payload)


def test_adaptive_diagnostics_reject_partial_or_unknown_fields() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        validate_diagnostics_payload({**_payload(), "lookahead_probability": 0.4})
    with pytest.raises(ValueError, match="keys differ"):
        validate_diagnostics_payload({**_payload(), "unrecognized_metric": 0.4})
