"""Read-only, training-proxy utility diagnosis; never a held-out control score.

Prepare a manifest first, then execute it unchanged. The current router was
trained on the full dataset, including the prospective validation partition.
"""

from __future__ import annotations

# Keep CPU manifest/statistics tests independent of the GPU training stack.
# ruff: noqa: PLC0415
import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest import mock

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key_hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def make_samples(episodes, tasks, *, split_seed: int, action_horizon: int):
    """Partition whole trajectories; choose midpoint with an unpadded chunk."""
    task_ids = {task["task"]: task["task_index"] for task in tasks}
    grouped = defaultdict(list)
    offset = 0
    seen = set()
    for ep in episodes:
        episode_id = int(ep["episode_index"])
        if episode_id in seen or len(ep["tasks"]) != 1 or ep["length"] <= 0:
            raise ValueError("Expected unique positive-length, single-task trajectories")
        seen.add(episode_id)
        # LeRobot stores trajectories contiguously in episode-index order.
        if episode_id != len(seen) - 1:
            raise ValueError("Episode metadata must be contiguous and ordered")
        grouped[task_ids[ep["tasks"][0]]].append({**ep, "offset": offset})
        offset += ep["length"]
    if set(grouped) != set(task_ids.values()):
        raise ValueError("Every task must have trajectories")
    samples, validation_ids = [], []
    for task_id, group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda ep: key_hash("split_v1", split_seed, task_id, ep["episode_index"]))
        n_val = math.ceil(0.1 * len(ranked))
        validation_ids.extend(ep["episode_index"] for ep in ranked[:n_val])
        for split, pool in (("development", ranked[n_val:]), ("prospective_validation", ranked[:n_val])):
            eligible = [ep for ep in pool if ep["length"] >= 2 * action_horizon]
            if not eligible:
                raise ValueError(f"No unpadded midpoint for task={task_id}, split={split}")
            ep = min(eligible, key=lambda ep: key_hash("sample_v1", split_seed, ep["episode_index"]))
            frame = ep["length"] // 2
            samples.append(
                {
                    "task_index": task_id,
                    "episode_index": ep["episode_index"],
                    "frame_index": frame,
                    "global_index": ep["offset"] + frame,
                    "episode_length": ep["length"],
                    "split": split,
                }
            )
    return samples, sorted(validation_ids), offset


def describe(values):
    x = np.asarray(values, dtype=float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Statistics require nonempty finite values")
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q10": float(np.quantile(x, 0.1)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.9)),
        "max": float(x.max()),
    }


def summarize(rows):
    """Evaluate soft targets and actual per-gate MSE, without fitting a router."""
    dev = [row for row in rows if row["split"] == "development"]
    if not dev:
        raise ValueError("Development rows required for constant target")
    dev_constant = float(np.mean([row["target"] for row in dev]))
    report = {
        "claim_boundary": "training-proxy diagnosis; current checkpoint saw both partitions",
        "development_mean_target": dev_constant,
        "splits": {},
    }
    for split in sorted({row["split"] for row in rows}):
        group = [row for row in rows if row["split"] == split]
        target = np.array([r["target"] for r in group])
        benefit = np.array([r["benefit"] for r in group])
        probabilities = {
            "router": [r["probability"] for r in group],
            "half": [0.5] * len(group),
            "dev_constant": [dev_constant] * len(group),
        }
        calibration = {}
        for name, values in probabilities.items():
            p = np.clip(np.array(values), 1e-7, 1 - 1e-7)
            calibration[name] = {
                "bce": float(np.mean(-target * np.log(p) - (1 - target) * np.log1p(-p))),
                "brier": float(np.mean((p - target) ** 2)),
                "sign_accuracy": float(np.mean((p >= 0.5) == (benefit >= 0))),
            }
        per_episode = defaultdict(list)
        for row in group:
            per_episode[row["episode_index"]].append(row)
        pairs = [pair for pair in per_episode.values() if len(pair) == 2]
        report["splits"][split] = {
            "rows": len(group),
            "episodes": len(per_episode),
            "benefit": describe(benefit),
            "positive_benefit_fraction": float(np.mean(benefit > 0)),
            "target": describe(target),
            "probability": describe(probabilities["router"]),
            "target_within_0.01_of_half": float(np.mean(np.abs(target - 0.5) < 0.01)),
            "two_draw_sign_agreement": float(np.mean([(p[0]["benefit"] >= 0) == (p[1]["benefit"] >= 0) for p in pairs]))
            if pairs
            else None,
            "calibration": calibration,
            "action_mse": {
                name: float(np.mean([r["loss_" + name] for r in group]))
                for name in ("none", "fixed", "half", "soft", "hard", "oracle_hard")
            },
        }
    return report


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def action_chunk_for_forward(actions, *, horizon, action_dim):
    """Match train_align_pytorch's explicit normalized-action float32 cast."""
    import torch

    if tuple(actions.shape) != (1, horizon, action_dim):
        raise ValueError("Unexpected action chunk shape")
    actions = actions.to(torch.float32)
    if not torch.isfinite(actions).all():
        raise ValueError("Actions must be finite")
    return actions


def half_fusion_velocity(adapter, projection, suffix, delta):
    """Use the real fusion/projection: reduced-precision matmul is not affine."""
    return projection(adapter.suffix_fusion(suffix, delta, residual_scale=0.5))


def source_paths():
    return [
        Path(__file__),
        Path("src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py"),
        Path("src/openpi/models_pytorch/scheme4_adaptive_lookahead.py"),
        Path("src/openpi/models_pytorch/preprocessing_pytorch.py"),
        Path("src/openpi/models_pytorch/pi0_align_pytorch.py"),
        Path("src/openpi/models_pytorch/delta_tokenizer.py"),
        Path("src/openpi/training/data_loader.py"),
        Path("src/openpi/training/config.py"),
        Path("src/openpi/transforms.py"),
        Path("src/openpi/policies/libero_policy.py"),
    ]


def prepare(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    meta = torch.load(args.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    samples, validation_ids, frames = make_samples(
        read_jsonl(args.dataset / "meta/episodes.jsonl"),
        read_jsonl(args.dataset / "meta/tasks.jsonl"),
        split_seed=args.split_seed,
        action_horizon=meta["config"]["model"]["action_horizon"],
    )
    paths = [
        *source_paths(),
        args.checkpoint / "model.safetensors",
        args.checkpoint / "metadata.pt",
        args.checkpoint / "assets/Sylvest/libero_plus_lerobot/norm_stats.json",
        args.dataset / "meta/info.json",
        args.dataset / "meta/episodes.jsonl",
        args.dataset / "meta/tasks.jsonl",
        Path(meta["config"]["scheme4_tokenizer_checkpoint"]),
    ]
    manifest = {
        "protocol": "scheme4_utility_diagnostic_v1",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split_seed": args.split_seed,
        "draws": 2,
        "claim_boundary": "NOT held-out for current checkpoint; prospective partition only",
        "preprocessing": "unchanged training augmentation; no teacher images needed",
        "total_frames": frames,
        "validation_episode_ids": validation_ids,
        "samples": samples,
        "hashes": {str(p): sha256(p) for p in paths},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {len(samples)} frames x 2 draws; {len(validation_ids)} prospective validation episodes", flush=True
    )


def run(output):
    import jax
    from safetensors.torch import load_model
    import torch

    from openpi.models.model import Observation
    from openpi.models_pytorch import pi0_proposal_variants_pytorch as proposal
    from openpi.training import checkpoints
    from openpi.training import config as config_module
    from openpi.training import data_loader

    manifest = json.loads((output / "manifest.json").read_text())
    if (output / "rows.jsonl").exists():
        raise FileExistsError("Refusing to overwrite diagnostic rows")
    for path, expected in manifest["hashes"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Manifest input changed: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for real-checkpoint diagnostic")
    checkpoint = Path(manifest["checkpoint"])
    saved = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=False)["config"]
    config = config_module.get_config("pi05_libero_plus_geo")
    primitives = {
        f.name: saved[f.name]
        for f in dataclasses.fields(config)
        if f.init and f.name in saved and isinstance(saved[f.name], (str, int, float, bool, type(None)))
    }
    config = dataclasses.replace(config, **primitives, model=dataclasses.replace(config.model, **saved["model"]))
    if config.data.repo_id != saved["data"]["repo_id"]:
        raise ValueError("Unexpected dataset config")
    if not config.scheme4_router_only_training or config.scheme4_training_intent_timestep != "t1_noise":
        raise ValueError("Diagnostic requires the predeclared router-only/t1 checkpoint")
    if any(
        (
            config.current_geo_loss_coeff,
            config.scheme4_delta_token_loss_coeff,
            config.scheme4_future_recon_loss_coeff,
            config.scheme4_condition_negative_loss_coeff,
        )
    ):
        raise ValueError("Teacher losses must already be zero in checkpoint metadata")
    factory_values = {
        f.name: saved["data"][f.name]
        for f in dataclasses.fields(config.data)
        if f.init and f.name in saved["data"] and isinstance(saved["data"][f.name], (str, int, float, bool, type(None)))
    }
    factory_values["future_teacher_obs_offset"] = None
    factory = dataclasses.replace(config.data, **factory_values)
    data_config = factory.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config, norm_stats=checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    )
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(raw_dataset, data_config)
    raw = raw_dataset
    while hasattr(raw, "_dataset"):
        raw = raw._dataset  # noqa: SLF001
    for sample in manifest["samples"]:
        meta_row = raw.hf_dataset[sample["global_index"]]
        for key in ("episode_index", "frame_index", "task_index"):
            if int(meta_row[key]) != sample[key]:
                raise ValueError(f"Raw dataset alignment changed: {key}")
    print("All sampled episode/frame/task indices verified against raw dataset", flush=True)
    if len(dataset) != manifest["total_frames"]:
        raise ValueError("Dataset length changed")
    torch.manual_seed(0)
    model = proposal.Scheme4LACDoubleAlignPI0Pytorch(config.model, config)
    load_model(model, checkpoint / "model.safetensors", strict=True)
    model.to("cuda").eval().requires_grad_(requires_grad=False)
    print("Strict checkpoint load passed; all parameters frozen", flush=True)
    device = torch.device("cuda")
    original_utility = proposal.utility_targets_from_losses
    rows = []
    started = time.perf_counter()
    with (output / "rows.jsonl").open("x") as stream, torch.no_grad():
        for sample in manifest["samples"]:
            for draw in range(manifest["draws"]):
                seed = int(key_hash("draw_v1", manifest["split_seed"], sample["global_index"], draw)[:15], 16)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                batch = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=device),
                    data_loader._collate_fn([dataset[sample["global_index"]]]),  # noqa: SLF001
                )
                obs = Observation.from_dict(batch)
                if obs.teacher_images is not None or obs.future_teacher_images is not None:
                    raise ValueError("Unexpected teacher image input")
                actions = action_chunk_for_forward(
                    batch["actions"], horizon=config.model.action_horizon, action_dim=config.model.action_dim
                )
                noise = model.sample_noise(actions.shape, device)
                timestep = model.sample_time(1, device)
                truth = noise - actions
                captured, projections, probabilities, fusion_inputs = {}, [], [], []

                def observe_utility(base, full, *, _captured=captured, **kwargs):
                    benefit, target = original_utility(base, full, **kwargs)
                    _captured.update(benefit=benefit.item(), target=target.item(), base=base.item(), full=full.item())
                    return benefit, target

                proj_hook = model.action_out_proj.register_forward_hook(
                    lambda _m, args, out, values=projections: values.append(
                        (args[0].detach().float().clone(), out.detach().float().clone())
                    )
                )
                fusion_hook = model.scheme4_delta_adapter.suffix_fusion.register_forward_hook(
                    lambda _m, args, _out, values=fusion_inputs: values.append((args[0].detach(), args[1].detach()))
                )
                router_hook = model.scheme4_utility_router.register_forward_hook(
                    lambda _m, _a, out, values=probabilities: values.append(out.item())
                )
                try:
                    with mock.patch.object(proposal, "utility_targets_from_losses", observe_utility):
                        metrics = model(obs, actions, vggt=None, noise=noise, time=timestep)
                finally:
                    proj_hook.remove()
                    router_hook.remove()
                    fusion_hook.remove()
                if len(projections) != 3 or len(probabilities) != 1 or len(fusion_inputs) != 2 or not captured:
                    raise ValueError("Forward observation contract changed")
                base_v, soft_v, full_v = [pair[1] for pair in projections]
                base_hidden, soft_hidden, full_hidden = [pair[0] for pair in projections]
                p = probabilities[0]

                def mse(velocity, truth=truth):
                    return (truth.float() - velocity).square().mean().item()

                if not math.isclose(mse(base_v), captured["base"], rel_tol=1e-5, abs_tol=1e-7) or not math.isclose(
                    mse(full_v), captured["full"], rel_tol=1e-5, abs_tol=1e-7
                ):
                    raise ValueError("Captured projections do not reproduce utility")
                # Verify gate semantics before the TF32-enabled output projection.
                torch.testing.assert_close(
                    soft_hidden, base_hidden + p * (full_hidden - base_hidden), rtol=1e-5, atol=1e-6
                )
                half_v = half_fusion_velocity(model.scheme4_delta_adapter, model.action_out_proj, *fusion_inputs[-1])
                row = {
                    **sample,
                    "draw": draw,
                    "seed": seed,
                    "flow_time": timestep.item(),
                    "noise_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    **captured,
                    "probability": p,
                    "loss_none": captured["base"],
                    "loss_fixed": captured["full"],
                    "loss_half": mse(half_v),
                    "loss_soft": mse(soft_v),
                    "loss_hard": captured["full"] if p >= 0.5 else captured["base"],
                    "loss_oracle_hard": min(captured["base"], captured["full"]),
                    "utility_bce": metrics["scheme4_utility_loss"].item(),
                    "velocity_interpolation_max_abs_error": (soft_v - (base_v + p * (full_v - base_v)))
                    .abs()
                    .max()
                    .item(),
                }
                if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError("Nonfinite diagnostic")
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 10 == 0:
                    print(f"{len(rows)}/{len(manifest['samples']) * manifest['draws']} forwards", flush=True)
    report = summarize(rows)
    report.update(
        elapsed_seconds=time.perf_counter() - started,
        gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        manifest_sha256=sha256(output / "manifest.json"),
        rows_sha256=sha256(output / "rows.jsonl"),
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-seed", type=int, default=20261110)
    args = parser.parse_args()
    if args.prepare:
        if args.checkpoint is None or args.dataset is None:
            parser.error("--prepare requires --checkpoint and --dataset")
        prepare(args)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
