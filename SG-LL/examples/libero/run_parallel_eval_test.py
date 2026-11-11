"""Focused tests for proposal-server override forwarding in parallel LIBERO evaluation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

MODULE_PATH = Path(__file__).with_name("run_parallel_eval.py")
SPEC = importlib.util.spec_from_file_location("run_parallel_eval_under_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_parallel_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_parallel_eval
SPEC.loader.exec_module(run_parallel_eval)


def test_condition_negative_mode_override_is_forwarded_only_when_explicit() -> None:
    server_cmd = ["python", "scripts/serve_policy_proposal.py"]

    run_parallel_eval.append_condition_negative_mode_override(
        server_cmd,
        argparse.Namespace(policy_scheme4_condition_negative_modes="default"),
    )
    assert server_cmd == ["python", "scripts/serve_policy_proposal.py"]

    run_parallel_eval.append_condition_negative_mode_override(
        server_cmd,
        argparse.Namespace(policy_scheme4_condition_negative_modes="language"),
    )
    assert server_cmd[-2:] == ["--policy-scheme4-condition-negative-modes", "language"]


def test_lookahead_mode_override_is_forwarded_only_when_explicit() -> None:
    server_cmd = ["python", "scripts/serve_policy_proposal.py"]
    run_parallel_eval.append_lookahead_mode_override(
        server_cmd,
        argparse.Namespace(policy_scheme4_lookahead_mode="default"),
    )
    assert server_cmd == ["python", "scripts/serve_policy_proposal.py"]

    run_parallel_eval.append_lookahead_mode_override(
        server_cmd,
        argparse.Namespace(policy_scheme4_lookahead_mode="soft"),
    )
    assert server_cmd[-2:] == ["--policy-scheme4-lookahead-mode", "soft"]


def test_policy_torch_seed_is_forwarded_only_when_explicit() -> None:
    server_cmd = ["python", "scripts/serve_policy_proposal.py"]
    run_parallel_eval.append_policy_torch_seed(server_cmd, argparse.Namespace(policy_torch_seed=None))
    assert server_cmd == ["python", "scripts/serve_policy_proposal.py"]

    run_parallel_eval.append_policy_torch_seed(server_cmd, argparse.Namespace(policy_torch_seed=7))
    assert server_cmd[-2:] == ["--policy-torch-seed", "7"]


def test_request_noise_seed_is_client_only_and_opt_in() -> None:
    client_cmd = ["python", "examples/libero/main.py"]
    run_parallel_eval.append_request_noise_seed(client_cmd, argparse.Namespace(request_noise_policy_seed=None))
    assert client_cmd == ["python", "examples/libero/main.py"]
    run_parallel_eval.append_request_noise_seed(client_cmd, argparse.Namespace(request_noise_policy_seed=7))
    assert client_cmd[-2:] == ["--request-noise-policy-seed", "7"]


def test_parallel_eval_cli_exposes_condition_negative_mode_override() -> None:
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--policy-scheme4-condition-negative-modes" in result.stdout
    assert "--policy-scheme4-lookahead-mode" in result.stdout
    assert "--policy-torch-seed" in result.stdout
    assert "--request-noise-policy-seed" in result.stdout
