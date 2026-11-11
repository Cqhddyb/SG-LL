#!/usr/bin/env python3
"""Launch sharded LIBERO evaluation clients against one policy server.

The policy server keeps one model copy on the selected GPU. Multiple client
processes evaluate disjoint task ranges and connect to that server, which
overlaps simulator work with policy inference without tripling model memory.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib import error
from urllib import request


def _split_range(task_start: int, num_tasks: int, num_workers: int) -> list[tuple[int, int]]:
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    num_workers = min(num_workers, num_tasks)
    shard_size = math.ceil(num_tasks / num_workers)
    shards = []
    for worker_idx in range(num_workers):
        start = task_start + worker_idx * shard_size
        stop = min(task_start + num_tasks, start + shard_size)
        if start < stop:
            shards.append((start, stop))
    return shards


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _wait_for_server(
    port: int,
    timeout_sec: float,
    *,
    process: subprocess.Popen | None = None,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_sec
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            log_tail = _tail_text(log_path) if log_path else ""
            raise RuntimeError(
                f"Policy server exited with code {process.returncode} before becoming healthy at {url}."
                + (f"\nLast server log lines:\n{log_tail}" if log_tail else "")
            )
        try:
            with request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, error.URLError):
            time.sleep(2)
    raise TimeoutError(f"Policy server did not become healthy at {url} within {timeout_sec:.0f}s")


def _env_with(base_env: dict[str, str], updates: dict[str, str | None]) -> dict[str, str]:
    env = dict(base_env)
    for key, value in updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _prepend_pythonpath(env: dict[str, str], path: str | None) -> None:
    if not path:
        return
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{path}:{current}" if current else path


def append_condition_negative_mode_override(server_cmd: list[str], args: argparse.Namespace) -> None:
    """Append the explicit Scheme4 negative-mode reconstruction override."""
    if args.policy_scheme4_condition_negative_modes != "default":
        server_cmd.extend(
            [
                "--policy-scheme4-condition-negative-modes",
                args.policy_scheme4_condition_negative_modes,
            ]
        )


def append_lookahead_mode_override(server_cmd: list[str], args: argparse.Namespace) -> None:
    """Forward an explicit Scheme4 lookahead mode without changing defaults."""
    if args.policy_scheme4_lookahead_mode != "default":
        server_cmd.extend(["--policy-scheme4-lookahead-mode", args.policy_scheme4_lookahead_mode])


def append_policy_torch_seed(server_cmd: list[str], args: argparse.Namespace) -> None:
    """Forward the legacy server-start RNG seed (alone it does not pair requests)."""
    if args.policy_torch_seed is not None:
        server_cmd.extend(["--policy-torch-seed", str(args.policy_torch_seed)])


def append_request_noise_seed(client_cmd: list[str], args: argparse.Namespace) -> None:
    """Explicitly opt clients into stable episode/request noise keys."""
    if args.request_noise_policy_seed is not None:
        client_cmd.extend(["--request-noise-policy-seed", str(args.request_noise_policy_seed)])


def _terminate_process(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        return
    print(f"[parallel-eval] terminating {name} pid={process.pid}", flush=True)
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print(f"[parallel-eval] killing {name} pid={process.pid}", flush=True)
        process.kill()
        process.wait(timeout=20)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", required=True, help="Name used for log/result directories.")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument(
        "--trial-start",
        type=int,
        default=0,
        help="First official initial-state index to evaluate per task; defaults to the original first state.",
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES value for the policy server.")
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument(
        "--policy-data-future-teacher-obs-offset",
        type=int,
        default=None,
        help="Optional proposal-server override for train_config.data.future_teacher_obs_offset.",
    )
    parser.add_argument(
        "--policy-scheme3-horizon-id",
        type=int,
        default=None,
        help="Optional proposal-server override for train_config.scheme3_horizon_id.",
    )
    parser.add_argument(
        "--policy-scheme4-delta-enabled",
        choices=("default", "true", "false"),
        default="default",
        help="Optional proposal-server override for train_config.scheme4_delta_enabled.",
    )
    parser.add_argument(
        "--policy-scheme4-training-action-condition",
        choices=("default", "normal", "zero_action"),
        default="default",
        help="Optional proposal-server override for train_config.scheme4_training_action_condition.",
    )
    parser.add_argument(
        "--policy-scheme4-condition-negative-modes",
        choices=("default", "language", "language,action"),
        default="default",
        help="Optional proposal-server override for train_config.scheme4_condition_negative_modes.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-condition-ablation",
        choices=(
            "default",
            "none",
            "zero_language",
            "zero_action",
            "zero_language_action",
            "wrong_language",
            "wrong_action",
            "shuffled_language",
            "shuffled_action",
        ),
        default="default",
        help="Optional proposal-server override for train_config.scheme4_inference_condition_ablation.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-delta-control",
        choices=(
            "default",
            "normal",
            "zero_delta",
            "mean_delta",
            "random_delta",
            "norm_matched_random_delta",
            "cross_episode_delta",
            "batch_shuffled_delta",
        ),
        default="default",
        help="Optional proposal-server override for train_config.scheme4_inference_delta_control.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-delta-mean-path",
        default=None,
        help="Optional .pt tensor path for train_config.scheme4_inference_delta_mean_path.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-control-seed",
        type=int,
        default=None,
        help="Optional proposal-server override for train_config.scheme4_inference_control_seed.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-fusion-scale",
        type=float,
        default=None,
        help="Optional proposal-server multiplier alpha in H' = H + alpha * delta_residual.",
    )
    parser.add_argument(
        "--policy-scheme4-inference-diagnostics-enabled",
        choices=("default", "true", "false"),
        default="default",
        help="Explicitly enable detached Scheme4 per-request diagnostics in proposal serving.",
    )
    parser.add_argument(
        "--policy-scheme4-lookahead-mode",
        choices=("default", "none", "fixed", "soft", "hard"),
        default="default",
        help="Explicit Scheme4 no/fixed/adaptive lookahead inference mode.",
    )
    parser.add_argument(
        "--policy-scheme4-tokenizer-checkpoint",
        default=None,
        help="Optional proposal-server override for train_config.scheme4_tokenizer_checkpoint.",
    )
    parser.add_argument(
        "--policy-torch-seed",
        type=int,
        default=None,
        help="Legacy server-start PyTorch RNG seed; does not pair individual requests.",
    )
    parser.add_argument(
        "--request-noise-policy-seed",
        type=int,
        default=None,
        help="Opt-in stable action noise keyed by suite/task/episode/env seed/policy seed/request index.",
    )
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--server-script",
        default="scripts/serve_policy.py",
        help="Policy server script to launch. Use scripts/serve_policy_proposal.py for proposal checkpoints.",
    )
    parser.add_argument(
        "--proposal-variant",
        choices=(
            "scheme1",
            "scheme1_rq",
            "scheme2",
            "scheme2_g",
            "scheme2_unified",
            "scheme2_unified_aux",
            "direct_vggt_injection",
            "scheme3_d3",
            "scheme3_delta",
            "scheme3_delta_intent",
            "scheme4_lac_double_align",
            "scheme4_grounded_lac_double_align",
            "scheme4_frame_token_lac_double_align",
        ),
        default=None,
        help="Proposal variant to serve. When set, the launcher uses scripts/serve_policy_proposal.py.",
    )
    parser.add_argument("--server-jax-platforms", default="cpu")
    parser.add_argument("--torchdynamo-disable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=600.0)

    parser.add_argument("--libero-pythonpath", default="../LIBERO-plus")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument(
        "--mujoco-egl-device-id",
        default=None,
        help="Optional MUJOCO_EGL_DEVICE_ID for client-side EGL rendering.",
    )
    parser.add_argument("--client-jax-platforms", default="cpu")
    parser.add_argument("--video-out-path", default=None)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-success-suffix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--result-dir", default="data/libero/results")
    parser.add_argument("--log-dir", default="logs/eval")
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Root for one diagnostics JSONL per worker. Requires diagnostics-enabled=true.",
    )

    args = parser.parse_args()
    if args.trial_start < 0:
        parser.error(f"--trial-start must be non-negative, got {args.trial_start}")
    if args.proposal_variant is not None and args.server_script == "scripts/serve_policy.py":
        args.server_script = "scripts/serve_policy_proposal.py"
    if Path(args.server_script).name == "serve_policy_proposal.py" and args.proposal_variant is None:
        raise SystemExit("--proposal-variant is required when using scripts/serve_policy_proposal.py")
    has_policy_overrides = (
        args.policy_data_future_teacher_obs_offset is not None
        or args.policy_scheme3_horizon_id is not None
        or args.policy_scheme4_delta_enabled != "default"
        or args.policy_scheme4_training_action_condition != "default"
        or args.policy_scheme4_condition_negative_modes != "default"
        or args.policy_scheme4_inference_condition_ablation != "default"
        or args.policy_scheme4_inference_delta_control != "default"
        or args.policy_scheme4_inference_delta_mean_path is not None
        or args.policy_scheme4_inference_control_seed is not None
        or args.policy_scheme4_inference_fusion_scale is not None
        or args.policy_scheme4_inference_diagnostics_enabled != "default"
        or args.policy_torch_seed is not None
        or args.policy_scheme4_tokenizer_checkpoint is not None
    )
    if has_policy_overrides and Path(args.server_script).name != "serve_policy_proposal.py":
        raise SystemExit("Policy config overrides are only supported with scripts/serve_policy_proposal.py")
    if (args.diagnostics_dir is not None) != (args.policy_scheme4_inference_diagnostics_enabled == "true"):
        raise SystemExit(
            "--diagnostics-dir and --policy-scheme4-inference-diagnostics-enabled=true must be set together"
        )

    repo_root = Path(__file__).resolve().parents[2]
    result_dir = Path(args.result_dir) / args.run_name
    log_dir = Path(args.log_dir) / args.run_name
    diagnostics_dir = Path(args.diagnostics_dir) / args.run_name if args.diagnostics_dir else None
    video_out_path = Path(args.video_out_path) if args.video_out_path else Path("data/libero/videos") / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
    video_out_path.mkdir(parents=True, exist_ok=True)

    shards = _split_range(args.task_start, args.num_tasks, args.num_workers)
    print(f"[parallel-eval] repo={repo_root}", flush=True)
    print(f"[parallel-eval] run={args.run_name} shards={shards}", flush=True)

    base_env = os.environ.copy()
    server_proc: subprocess.Popen | None = None
    server_log = None
    client_procs: list[tuple[str, subprocess.Popen, object]] = []

    try:
        if args.start_server:
            server_log_path = log_dir / "server.log"
            server_log = server_log_path.open("w", encoding="utf-8")
            server_env = _env_with(
                base_env,
                {
                    "CUDA_VISIBLE_DEVICES": args.gpu,
                    "JAX_PLATFORMS": args.server_jax_platforms,
                    "PYTORCH_CUDA_ALLOC_CONF": base_env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                    "TORCHDYNAMO_DISABLE": "1" if args.torchdynamo_disable else None,
                },
            )
            server_cmd = [
                sys.executable,
                args.server_script,
            ]
            if args.proposal_variant is not None:
                server_cmd.extend(["--proposal-variant", args.proposal_variant])
            if args.policy_data_future_teacher_obs_offset is not None:
                server_cmd.extend(
                    [
                        "--policy-data-future-teacher-obs-offset",
                        str(args.policy_data_future_teacher_obs_offset),
                    ]
                )
            if args.policy_scheme3_horizon_id is not None:
                server_cmd.extend(["--policy-scheme3-horizon-id", str(args.policy_scheme3_horizon_id)])
            if args.policy_scheme4_delta_enabled != "default":
                server_cmd.extend(["--policy-scheme4-delta-enabled", args.policy_scheme4_delta_enabled])
            if args.policy_scheme4_training_action_condition != "default":
                server_cmd.extend(
                    [
                        "--policy-scheme4-training-action-condition",
                        args.policy_scheme4_training_action_condition,
                    ]
                )
            append_condition_negative_mode_override(server_cmd, args)
            if args.policy_scheme4_inference_condition_ablation != "default":
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-condition-ablation",
                        args.policy_scheme4_inference_condition_ablation,
                    ]
                )
            if args.policy_scheme4_inference_delta_control != "default":
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-delta-control",
                        args.policy_scheme4_inference_delta_control,
                    ]
                )
            if args.policy_scheme4_inference_delta_mean_path is not None:
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-delta-mean-path",
                        args.policy_scheme4_inference_delta_mean_path,
                    ]
                )
            if args.policy_scheme4_inference_control_seed is not None:
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-control-seed",
                        str(args.policy_scheme4_inference_control_seed),
                    ]
                )
            if args.policy_scheme4_inference_fusion_scale is not None:
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-fusion-scale",
                        str(args.policy_scheme4_inference_fusion_scale),
                    ]
                )
            if args.policy_scheme4_inference_diagnostics_enabled != "default":
                server_cmd.extend(
                    [
                        "--policy-scheme4-inference-diagnostics-enabled",
                        args.policy_scheme4_inference_diagnostics_enabled,
                    ]
                )
            append_lookahead_mode_override(server_cmd, args)
            append_policy_torch_seed(server_cmd, args)
            if args.policy_scheme4_tokenizer_checkpoint is not None:
                server_cmd.extend(
                    [
                        "--policy-scheme4-tokenizer-checkpoint",
                        args.policy_scheme4_tokenizer_checkpoint,
                    ]
                )
            server_cmd.extend(
                [
                    "--env",
                    "LIBERO",
                    "--port",
                    str(args.port),
                    "policy:checkpoint",
                    "--policy.config",
                    args.policy_config,
                    "--policy.dir",
                    args.policy_dir,
                ]
            )
            print(f"[parallel-eval] starting server: {' '.join(server_cmd)}", flush=True)
            print(f"[parallel-eval] server log: {server_log_path}", flush=True)
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=repo_root,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for_server(args.port, args.server_ready_timeout_sec, process=server_proc, log_path=server_log_path)
            print(f"[parallel-eval] server healthy on port {args.port}", flush=True)
        else:
            _wait_for_server(args.port, 30.0)
            print(f"[parallel-eval] using existing server on port {args.port}", flush=True)

        for shard_idx, (start, stop) in enumerate(shards):
            shard_tasks = stop - start
            shard_name = f"shard{shard_idx:02d}_tasks{start:04d}_{stop - 1:04d}"
            result_path = result_dir / f"{shard_name}.jsonl"
            client_log_path = log_dir / f"{shard_name}.log"
            client_log = client_log_path.open("w", encoding="utf-8")
            client_env = _env_with(
                base_env,
                {
                    "JAX_PLATFORMS": args.client_jax_platforms,
                    "MUJOCO_GL": args.mujoco_gl,
                    "MUJOCO_EGL_DEVICE_ID": args.mujoco_egl_device_id,
                },
            )
            _prepend_pythonpath(client_env, args.libero_pythonpath)
            client_cmd = [
                sys.executable,
                "examples/libero/main.py",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--task-suite-name",
                args.task_suite_name,
                "--num-trials-per-task",
                str(args.num_trials_per_task),
                "--trial-start",
                str(args.trial_start),
                "--task-start",
                str(start),
                "--num-tasks",
                str(shard_tasks),
                "--seed",
                str(args.seed),
                "--video-out-path",
                str(video_out_path),
                "--result-log-path",
                str(result_path),
            ]
            append_request_noise_seed(client_cmd, args)
            if diagnostics_dir is not None:
                client_cmd.extend(
                    [
                        "--diagnostics-log-path",
                        str(diagnostics_dir / f"{shard_name}.jsonl"),
                    ]
                )
            if not args.save_videos:
                client_cmd.append("--no-save-videos")
            if args.video_success_suffix:
                client_cmd.append("--video-success-suffix")

            print(f"[parallel-eval] starting {shard_name}: {' '.join(client_cmd)}", flush=True)
            print(f"[parallel-eval] {shard_name} log: {client_log_path}", flush=True)
            process = subprocess.Popen(
                client_cmd,
                cwd=repo_root,
                env=client_env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            client_procs.append((shard_name, process, client_log))

        failed = []
        for shard_name, process, _client_log in client_procs:
            return_code = process.wait()
            print(f"[parallel-eval] {shard_name} exited with code {return_code}", flush=True)
            if return_code != 0:
                failed.append((shard_name, return_code))

        if failed:
            raise SystemExit(f"Some shards failed: {failed}")
        print(f"[parallel-eval] all shards complete. Results: {result_dir}", flush=True)
        print(f"[parallel-eval] logs: {log_dir}", flush=True)

    except KeyboardInterrupt:
        print("[parallel-eval] interrupted", flush=True)
        for shard_name, process, _client_log in client_procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            _terminate_process(process, shard_name)
        raise
    finally:
        for _shard_name, _process, client_log in client_procs:
            client_log.close()
        if server_proc is not None:
            _terminate_process(server_proc, "server")
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
