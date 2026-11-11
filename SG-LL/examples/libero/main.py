import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import time

import imageio
from initial_state_window import resolve_initial_state_indices
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from openpi.shared.request_noise import NOISE_PROTOCOL
from openpi.shared.request_noise import NOISE_SEED_KEY
from openpi.shared.request_noise import episode_request_seed
from openpi.shared.scheme4_diagnostics_trace import make_diagnostics_row

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    trial_start: int = 0  # First official initial-state index to evaluate per task.
    task_start: int = 0  # First task id to evaluate.
    num_tasks: int | None = None  # If set, evaluate only this many tasks.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True  # Save the latest replay video for each task.
    video_success_suffix: bool = False  # If true, keep separate success/failure videos instead of overwriting latest.
    result_log_path: str | None = None  # JSONL episode log. If unset, a timestamped file is created.
    diagnostics_log_path: str | None = None  # Optional per-policy-request Scheme4 diagnostic JSONL.

    seed: int = 7  # Random Seed (for reproducibility)
    request_noise_policy_seed: int | None = None  # Opt-in per-episode/request action noise; legacy default unchanged.


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    if args.task_start < 0 or args.task_start >= num_tasks_in_suite:
        raise ValueError(f"task_start must be in [0, {num_tasks_in_suite}), got {args.task_start}")
    task_stop = (
        num_tasks_in_suite if args.num_tasks is None else min(num_tasks_in_suite, args.task_start + args.num_tasks)
    )
    task_ids = range(args.task_start, task_stop)
    logging.info(f"Evaluating task ids [{args.task_start}, {task_stop}) out of {num_tasks_in_suite}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    result_log_path = _resolve_result_log_path(args, task_stop)
    diagnostics_log_path = pathlib.Path(args.diagnostics_log_path) if args.diagnostics_log_path else None
    if diagnostics_log_path is not None:
        diagnostics_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id(args)
    logging.info(f"Writing episode result JSONL to: {result_log_path}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        trial_indices = resolve_initial_state_indices(
            trial_start=args.trial_start,
            num_trials_per_task=args.num_trials_per_task,
            num_initial_states=len(initial_states),
        )

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(trial_indices):
            logging.info(f"\nTask: {task_description}")
            episode_start_time = time.monotonic()
            error_message = None

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            policy_request_index = 0

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        request_seed = None
                        if args.request_noise_policy_seed is not None:
                            request_seed = episode_request_seed(
                                suite=args.task_suite_name,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                env_seed=int(args.seed),
                                policy_seed=int(args.request_noise_policy_seed),
                                request_index=policy_request_index,
                            )
                            element[NOISE_SEED_KEY] = request_seed
                        # Query model to get action
                        response = client.infer(element)
                        if request_seed is not None:
                            acknowledgement = response.get("action_noise", {})
                            if (
                                acknowledgement.get("seed") != request_seed
                                or acknowledgement.get("protocol") != NOISE_PROTOCOL
                                or len(acknowledgement.get("sha256", "")) != 64
                            ):
                                raise RuntimeError(
                                    "Policy did not acknowledge the requested action noise protocol/seed"
                                )
                        action_chunk = response["actions"]
                        if diagnostics_log_path is not None:
                            diagnostics = response.get("scheme4_inference_diagnostics")
                            if not isinstance(diagnostics, dict):
                                raise RuntimeError(
                                    "Diagnostics logging was requested but the policy response has no "
                                    "scheme4_inference_diagnostics object"
                                )
                            diagnostics_row = make_diagnostics_row(
                                run_id=run_id,
                                suite=args.task_suite_name,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                seed=int(args.seed),
                                policy_request_index=policy_request_index,
                                env_step=int(t),
                                replan_steps=int(args.replan_steps),
                                diagnostics=diagnostics,
                            )
                            if request_seed is not None:
                                diagnostics_row["action_noise"] = response["action_noise"]
                                diagnostics_row["action_chunk_sha256"] = hashlib.sha256(
                                    np.ascontiguousarray(action_chunk).tobytes()
                                ).hexdigest()
                            _append_jsonl(diagnostics_log_path, diagnostics_row)
                        policy_request_index += 1
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    error_message = repr(e)
                    logging.error(f"Caught exception: {e}")
                    break

            global_episode = total_episodes + 1
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the latest episode. By default this overwrites prior videos for the same task.
            task_segment = task_description.replace(" ", "_")
            if args.video_success_suffix:
                suffix = "success" if done else "failure"
                video_filename = f"rollout_{task_segment}_{suffix}.mp4"
            else:
                video_filename = f"rollout_{task_segment}.mp4"
            video_path = pathlib.Path(args.video_out_path) / video_filename
            if args.save_videos and replay_images:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            elif not replay_images:
                logging.warning(
                    "No replay images captured; skipping video write for task_id=%s episode=%s", task_id, episode_idx
                )

            result = {
                "type": "episode",
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "suite": args.task_suite_name,
                "task_id": task_id,
                "classification_task_id": task_id + 1,
                "global_episode": global_episode,
                "episode_idx": episode_idx,
                "success": bool(done),
                "steps": int(t),
                "max_steps": int(max_steps),
                "num_steps_wait": int(args.num_steps_wait),
                "num_trials_per_task": int(args.num_trials_per_task),
                "trial_start": int(args.trial_start),
                "trial_stop": int(args.trial_start + args.num_trials_per_task),
                "task_start": int(args.task_start),
                "task_stop": int(task_stop),
                "seed": int(args.seed),
                "task": str(task_description),
                "video_path": str(video_path) if args.save_videos else "",
                "error": error_message,
                "elapsed_sec": time.monotonic() - episode_start_time,
                "degradation": _degradation_env_snapshot(),
            }
            if diagnostics_log_path is not None:
                result["policy_requests"] = int(policy_request_index)
                result["replan_steps"] = int(args.replan_steps)
            if args.request_noise_policy_seed is not None:
                result["action_noise_protocol"] = NOISE_PROTOCOL
                result["request_noise_policy_seed"] = int(args.request_noise_policy_seed)
                result["policy_requests"] = int(policy_request_index)
            _append_jsonl(result_log_path, result)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(
                "Saved rollout MP4 at path %s --episode=%d--success=%s--task=%s",
                result["video_path"],
                result["classification_task_id"],
                str(bool(done)),
                task_segment,
            )
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        try:
            env.close()
        except Exception as e:
            logging.warning(f"Failed to close LIBERO env cleanly: {e}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Episode result JSONL: {result_log_path}")


def _make_run_id(args: Args) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.task_suite_name}_start{args.task_start}_{timestamp}"


def _resolve_result_log_path(args: Args, task_stop: int) -> pathlib.Path:
    if args.result_log_path is not None:
        path = pathlib.Path(args.result_log_path)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.task_suite_name}_tasks{args.task_start:04d}_{task_stop - 1:04d}_{timestamp}.jsonl"
        path = pathlib.Path("data/libero/results") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _degradation_env_snapshot() -> dict[str, str]:
    keys = (
        "LIBERO_IMAGE_DEGRADATION",
        "LIBERO_IMAGE_DEGRADATION_SEED",
        "LIBERO_IMAGE_DEGRADATION_EPISODE_SEED",
        "LIBERO_IMAGE_DEGRADATION_APPLY_TO",
        "LIBERO_IMAGE_DEGRADATION_VIEW_PROBS",
        "LIBERO_IMAGE_DEGRADATION_OP_PROBS",
        "LIBERO_IMAGE_DEGRADATION_VIEW_DEFAULT_PROB",
        "LIBERO_IMAGE_DEGRADATION_OP_DEFAULT_PROB",
        "LIBERO_IMAGE_DEGRADATION_DETERMINISTIC",
        "LIBERO_SPOT_NUM_CLEAN_VIEWS",
        "LIBERO_SPOT_CLEAN_KEYS",
    )
    return {key: os.environ.get(key, "") for key in keys}


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
