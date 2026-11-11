"""One predeclared frozen-feature utility experiment; not a control benchmark.

Only router utility training excludes validation trajectories. The frozen VLA
has seen them. No cache is written: pooled current inputs stay in RAM.
"""

from __future__ import annotations

# Keep manifest/statistics tests independent of the VLA dependency stack.
# ruff: noqa: PLC0415
import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from openpi.models_pytorch.scheme4_adaptive_lookahead import Scheme4UtilityRouter
from scripts.diagnose_scheme4_utility import action_chunk_for_forward
from scripts.diagnose_scheme4_utility import key_hash
from scripts.diagnose_scheme4_utility import make_samples
from scripts.diagnose_scheme4_utility import read_jsonl
from scripts.diagnose_scheme4_utility import sha256
from scripts.diagnose_scheme4_utility import source_paths


def select_samples(episodes, tasks, prior, *, action_horizon, smoke=False):
    """Select by hash, excluding ALL reserved episodes from router fitting."""
    _, reserved, frames = make_samples(episodes, tasks, split_seed=prior["split_seed"], action_horizon=action_horizon)
    if reserved != prior["validation_episode_ids"] or frames != prior["total_frames"]:
        raise ValueError("Prospective split changed")
    reserved = set(reserved)
    previously_evaluated = {s["episode_index"] for s in prior["samples"] if s["split"] == "prospective_validation"}
    task_ids = {t["task"]: t["task_index"] for t in tasks}
    grouped = defaultdict(list)
    offset = 0
    for ep in episodes:
        if ep["length"] >= 2 * action_horizon:
            grouped[task_ids[ep["tasks"][0]]].append({**ep, "offset": offset})
        offset += ep["length"]
    samples = []
    task_indices = sorted(grouped)[:1] if smoke else sorted(grouped)
    for task_id in task_indices:
        ranked = sorted(
            grouped[task_id],
            key=lambda ep: key_hash("independent_utility_v1", prior["split_seed"], ep["episode_index"]),
        )
        for split, count in (("train", 1 if smoke else 8), ("validation", 1 if smoke else 2)):
            pool = [
                ep
                for ep in ranked
                if (ep["episode_index"] in reserved) == (split == "validation")
                and (split != "validation" or ep["episode_index"] not in previously_evaluated)
            ]
            if len(pool) < count:
                raise ValueError(f"Insufficient eligible episodes for {task_id}/{split}")
            for ep in pool[:count]:
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
    verify_isolation(samples, reserved)
    return samples


def verify_isolation(samples, reserved):
    train = {s["episode_index"] for s in samples if s["split"] == "train"}
    validation = {s["episode_index"] for s in samples if s["split"] == "validation"}
    if not train or not validation or train & set(reserved) or not validation <= set(reserved):
        raise ValueError("Whole-trajectory train/validation isolation failed")
    if any(s["split"] not in ("train", "validation") for s in samples):
        raise ValueError("Unknown split")


def pooled_inputs(args, kwargs):
    """Router's masked spatial mean [B,1,D]; other current inputs [B,D]."""
    spatial = args[0].detach().float()
    mask = kwargs["token_mask"]
    if mask is None:
        summary = spatial.mean(dim=1)
    else:
        mask = mask.to(device=spatial.device, dtype=spatial.dtype).unsqueeze(-1)
        summary = (spatial * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return {
        "spatial_tokens": summary.unsqueeze(1).cpu(),
        **{key: kwargs[key].detach().float().cpu() for key in ("language_features", "state", "action_intent")},
    }


def router_forward(router, features):
    return router(**features, token_mask=None)


def utility_coefficients(truth, base, full):
    if truth.shape != base.shape or base.shape != full.shape or base.ndim != 3:
        raise ValueError("Velocities and truth must share shape [B,T,A]")
    if not all(torch.isfinite(t).all() for t in (truth, base, full)):
        raise ValueError("Velocities must be finite")
    error, delta = truth.detach().float() - base.detach().float(), full.detach().float() - base.detach().float()
    return (delta.square().mean((1, 2))).detach(), ((error * delta).mean((1, 2))).detach()


def utility_risk(gate, q, c):
    if gate.ndim != 1 or gate.shape != q.shape or gate.shape != c.shape:
        raise ValueError("Gate and coefficients must share shape [B]")
    if not all(torch.isfinite(t).all() for t in (gate, q, c)) or (q < 0).any():
        raise ValueError("Nonfinite gate/coefficients or negative curvature")
    return q * gate.square() - 2 * c * gate


def constant_gate(q, c):
    if q.numel() == 0 or q.shape != c.shape:
        raise ValueError("Nonempty matched coefficient vectors required")
    utility_risk(torch.zeros_like(q), q, c)
    return float((c.mean() / q.mean()).clamp(0, 1)) if q.mean() > 0 else 0.0


def fit_router(router, features, q, c, *, settings, normalization):
    """No validation argument: fixed budget, fresh optimizer, train rows only."""
    if normalization <= 0 or not np.isfinite(normalization):
        raise ValueError("Positive finite training normalization required")
    if any(value.requires_grad for value in (*features.values(), q, c)):
        raise ValueError("Cached current inputs and supervised targets must be detached")
    if any(value.shape[0] != q.numel() for value in features.values()):
        raise ValueError("Feature rows must align with coefficients")
    router.train().requires_grad_(requires_grad=True)
    optimizer = torch.optim.AdamW(router.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(settings["seed"])
    trace = []
    for step in range(settings["steps"]):
        indices = torch.randint(q.numel(), (min(settings["batch_size"], q.numel()),), generator=generator)
        indices = indices.to(q.device)
        lr = settings["lr"] * min(1.0, (step + 1) / settings["warmup_steps"])
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        probability = router_forward(router, {key: value[indices] for key, value in features.items()})
        loss = utility_risk(probability, q[indices], c[indices]).mean() / normalization
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters()):
            raise ValueError("Missing or nonfinite router gradient")
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == settings["steps"]:
            trace.append({"step": step + 1, "normalized_risk": loss.item(), "lr": lr})
    router.eval().requires_grad_(requires_grad=False)
    return trace


def paired_interval(rows, control, *, seed, repeats=10000):
    """Pair within episode first; episode bootstrap is NOT training-seed variance."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_index"]].append(row["loss_" + control] - row["loss_soft"])
    differences = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    if differences.size == 0:
        raise ValueError("Paired rows required")
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(repeats, len(differences)))].mean(axis=1)
    return {
        "episodes": len(differences),
        "mean_control_minus_soft": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


def prepare(args):
    prior = json.loads((args.prior / "manifest.json").read_text())
    # The prior script hashes may have changed later; data and checkpoint must not.
    paths = [Path(prior["checkpoint"]) / name for name in ("model.safetensors", "metadata.pt")]
    paths.extend(Path(prior["dataset"]) / "meta" / name for name in ("episodes.jsonl", "tasks.jsonl", "info.json"))
    for path in paths:
        if sha256(path) != prior["hashes"][str(path)]:
            raise ValueError(f"Prior data/checkpoint changed: {path}")
    saved = torch.load(Path(prior["checkpoint"]) / "metadata.pt", map_location="cpu", weights_only=False)["config"]
    samples = select_samples(
        read_jsonl(Path(prior["dataset"]) / "meta/episodes.jsonl"),
        read_jsonl(Path(prior["dataset"]) / "meta/tasks.jsonl"),
        prior,
        action_horizon=saved["model"]["action_horizon"],
        smoke=args.smoke,
    )
    paths.extend([*source_paths(), Path(__file__), args.prior / "manifest.json"])
    paths.extend(
        [
            Path(prior["checkpoint"]) / "assets/Sylvest/libero_plus_lerobot/norm_stats.json",
            Path(saved["scheme4_tokenizer_checkpoint"]),
            Path("docs/decisions/2026-11-10_independent_utility_gate.md"),
        ]
    )
    manifest = {
        "protocol": "scheme4_independent_utility_gate_v1",
        "claim_boundary": "router-utility trajectory exclusion only; pretrained VLA saw validation",
        "checkpoint": prior["checkpoint"],
        "dataset": prior["dataset"],
        "split_seed": prior["split_seed"],
        "validation_episode_ids": prior["validation_episode_ids"],
        "samples": samples,
        "total_frames": prior["total_frames"],
        "draws": 2,
        "smoke": args.smoke,
        "training": {
            "seed": 42,
            "steps": 2 if args.smoke else 200,
            "batch_size": 64,
            "lr": 5e-5,
            "warmup_steps": 50,
            "weight_decay": 0.01,
        },
        "hashes": {str(path): sha256(path) for path in paths},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing manifest")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(samples)} trajectories x2 draws; whole-trajectory isolation passed")


def load_frozen(manifest):
    from safetensors.torch import load_model

    from openpi.models_pytorch.pi0_proposal_variants_pytorch import Scheme4LACDoubleAlignPI0Pytorch
    from openpi.training import checkpoints
    from openpi.training import config as config_module
    from openpi.training import data_loader

    checkpoint = Path(manifest["checkpoint"])
    saved = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=False)["config"]
    config = config_module.get_config("pi05_libero_plus_geo")
    primitives = {
        f.name: saved[f.name]
        for f in dataclasses.fields(config)
        if f.init and f.name in saved and isinstance(saved[f.name], (str, int, float, bool, type(None)))
    }
    config = dataclasses.replace(config, **primitives, model=dataclasses.replace(config.model, **saved["model"]))
    if not config.scheme4_router_only_training or config.scheme4_training_intent_timestep != "t1_noise":
        raise ValueError("Expected t1-noise router checkpoint")
    if any(
        (
            config.current_geo_loss_coeff,
            config.scheme4_delta_token_loss_coeff,
            config.scheme4_future_recon_loss_coeff,
            config.scheme4_condition_negative_loss_coeff,
        )
    ):
        raise ValueError("All teacher losses must be zero")
    factory_values = {
        f.name: saved["data"][f.name]
        for f in dataclasses.fields(config.data)
        if f.init and f.name in saved["data"] and isinstance(saved["data"][f.name], (str, int, float, bool, type(None)))
    }
    factory = dataclasses.replace(config.data, **{**factory_values, "future_teacher_obs_offset": None})
    data_config = factory.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config, norm_stats=checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    )
    raw = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(raw, data_config)
    while hasattr(raw, "_dataset"):
        raw = raw._dataset  # noqa: SLF001
    if len(dataset) != manifest["total_frames"]:
        raise ValueError("Dataset size changed")
    for row in manifest["samples"]:
        meta = raw.hf_dataset[row["global_index"]]
        if any(int(meta[key]) != row[key] for key in ("episode_index", "frame_index", "task_index")):
            raise ValueError("Raw dataset alignment changed")
    print("All sampled raw episode/frame/task indices verified", flush=True)
    model = Scheme4LACDoubleAlignPI0Pytorch(config.model, config)
    load_model(model, checkpoint / "model.safetensors", strict=True)
    old_router = model.scheme4_utility_router
    torch.manual_seed(manifest["training"]["seed"])
    dims = {
        key: getattr(old_router, key)
        for key in ("spatial_dim", "language_dim", "state_dim", "action_intent_dim", "hidden_dim")
    }
    model.scheme4_utility_router = Scheme4UtilityRouter(**dims)
    model.to("cuda").eval().requires_grad_(requires_grad=False)
    print("Strict load passed; router reinitialized; all modules frozen", flush=True)
    return model, dataset, dims


def seeded_forward(model, dataset, sample, draw, split_seed):
    import jax

    from openpi.models.model import Observation
    from openpi.training import data_loader

    seed = int(key_hash("independent_draw_v1", split_seed, sample["global_index"], draw)[:15], 16)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    batch = jax.tree.map(
        lambda x: torch.as_tensor(x, device="cuda"),
        data_loader._collate_fn([dataset[sample["global_index"]]]),  # noqa: SLF001
    )
    obs = Observation.from_dict(batch)
    if obs.teacher_images is not None or obs.future_teacher_images is not None:
        raise ValueError("Teacher inputs must be absent")
    actions = action_chunk_for_forward(
        batch["actions"], horizon=model.config.action_horizon, action_dim=model.config.action_dim
    )
    noise, timestep = (
        model.sample_noise(actions.shape, torch.device("cuda")),
        model.sample_time(1, torch.device("cuda")),
    )
    captured = {"projections": [], "features": [], "probability": [], "fusion_inputs": []}

    # Use the original forward, avoiding recursion through its observation hook.
    original_router_forward = model.scheme4_utility_router.forward

    def observe_without_hook(module, args, kwargs, result):
        pooled = pooled_inputs(args, kwargs)
        replay = original_router_forward(**{k: v.to("cuda") for k, v in pooled.items()}, token_mask=None)
        torch.testing.assert_close(replay, result, rtol=1e-6, atol=1e-7)
        captured["features"].append(pooled)
        captured["probability"].append(result.item())

    handles = [
        model.scheme4_utility_router.register_forward_hook(observe_without_hook, with_kwargs=True),
        model.action_out_proj.register_forward_hook(
            lambda _m, _a, out: captured["projections"].append(out.detach().float())
        ),
        model.scheme4_delta_adapter.suffix_fusion.register_forward_hook(
            lambda _m, args, _out: captured["fusion_inputs"].append((args[0].detach(), args[1].detach()))
        ),
    ]
    try:
        with torch.no_grad():
            model(obs, actions, vggt=None, noise=noise, time=timestep)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured["projections"]) != 3 or len(captured["features"]) != 1 or len(captured["fusion_inputs"]) != 2:
        raise ValueError("Observed forward contract changed")
    captured.update(
        truth=noise - actions,
        seed=seed,
        flow_time=timestep.item(),
        noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
    )
    return captured


def evaluate_row(model, captured, constants):
    truth = captured["truth"].float()
    base, soft, full = captured["projections"]

    def mse(velocity):
        return (truth - velocity).square().mean().item()

    p = captured["probability"][0]
    q, c = utility_coefficients(truth, base, full)
    row = {
        "probability": p,
        "q": q.item(),
        "c": c.item(),
        "loss_none": mse(base),
        "loss_fixed": mse(full),
        "loss_soft": mse(soft),
        "loss_hard": mse(full if p >= 0.5 else base),
        "affine_soft_mse": mse(base) + utility_risk(torch.tensor([p], device=q.device), q, c).item(),
    }
    with torch.no_grad():
        for key, gate in {"half": 0.5, **constants}.items():
            hidden = model.scheme4_delta_adapter.suffix_fusion(*captured["fusion_inputs"][-1], residual_scale=gate)
            row["loss_" + key] = mse(model.action_out_proj(hidden))
    return row


def run(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if (output / "train_rows.jsonl").exists() or (output / "router.pt").exists():
        raise FileExistsError("Refusing to overwrite experiment")
    for path, expected in manifest["hashes"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Manifest input changed: {path}")
    verify_isolation(manifest["samples"], manifest["validation_episode_ids"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    started = time.perf_counter()
    model, dataset, dims = load_frozen(manifest)
    inherited_versions = {
        name: p._version  # noqa: SLF001
        for name, p in model.named_parameters()
        if not name.startswith("scheme4_utility_router.")
    }
    train_features, train_rows = [], []
    with (output / "train_rows.jsonl").open("x") as stream:
        for sample in (s for s in manifest["samples"] if s["split"] == "train"):
            for draw in range(manifest["draws"]):
                captured = seeded_forward(model, dataset, sample, draw, manifest["split_seed"])
                base, _, full = captured["projections"]
                q, c = utility_coefficients(captured["truth"], base, full)
                row = {
                    **sample,
                    "draw": draw,
                    **{k: captured[k] for k in ("seed", "flow_time", "noise_sha256")},
                    "q": q.item(),
                    "c": c.item(),
                    "loss_none": (captured["truth"] - base).square().mean().item(),
                    "loss_fixed": (captured["truth"] - full).square().mean().item(),
                }
                train_features.append(captured["features"][0])
                train_rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(train_rows) % 40 == 0:
                    print(f"Collected {len(train_rows)} training rows", flush=True)
    features = {key: torch.cat([f[key] for f in train_features]).to("cuda") for key in train_features[0]}
    del train_features
    q = torch.tensor([r["q"] for r in train_rows], device="cuda")
    c = torch.tensor([r["c"] for r in train_rows], device="cuda")
    normalization = float(np.mean([r["loss_none"] for r in train_rows]))
    constants = {
        "train_optimal": constant_gate(q, c),
        "train_mean_target": float(
            torch.sigmoid(
                (torch.tensor([r["loss_none"] - r["loss_fixed"] for r in train_rows]) - model.scheme4_utility_margin)
                / model.scheme4_utility_temperature
            ).mean()
        ),
    }
    trace = fit_router(
        model.scheme4_utility_router, features, q, c, settings=manifest["training"], normalization=normalization
    )
    with torch.no_grad():
        probabilities = router_forward(model.scheme4_utility_router, features)
        training_risk = utility_risk(probabilities, q, c).mean().item()
    (output / "training.json").write_text(
        json.dumps(
            {"trace": trace, "normalization": normalization, "constants": constants, "final_risk": training_risk},
            indent=2,
        )
        + "\n"
    )
    torch.save(
        {
            "router_state_dict": {k: v.cpu() for k, v in model.scheme4_utility_router.state_dict().items()},
            "dimensions": dims,
            "training": manifest["training"],
            "manifest_sha256": sha256(output / "manifest.json"),
            "train_rows_sha256": sha256(output / "train_rows.jsonl"),
            "constants": constants,
        },
        output / "router.pt",
    )
    restored = Scheme4UtilityRouter(**dims).to("cuda")
    restored.load_state_dict(torch.load(output / "router.pt", weights_only=True)["router_state_dict"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(router_forward(restored, features), probabilities, rtol=0, atol=0)
    del features, restored, probabilities
    print("Training complete; router saved and strict reload verified BEFORE validation", flush=True)
    rows = []
    with (output / "validation_rows.jsonl").open("x") as stream:
        for sample in (s for s in manifest["samples"] if s["split"] == "validation"):
            for draw in range(manifest["draws"]):
                captured = seeded_forward(model, dataset, sample, draw, manifest["split_seed"])
                row = {
                    **sample,
                    "draw": draw,
                    **{k: captured[k] for k in ("seed", "flow_time", "noise_sha256")},
                    **evaluate_row(model, captured, constants),
                }
                if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError("Nonfinite validation row")
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 40 == 0:
                    print(f"Validated {len(rows)} rows", flush=True)
    for name, p in model.named_parameters():
        if p.requires_grad or (name in inherited_versions and p._version != inherited_versions[name]):  # noqa: SLF001
            raise ValueError(f"Frozen boundary changed: {name}")
    paired = {
        key: paired_interval(rows, key, seed=manifest["split_seed"]) for key in ("train_optimal", "fixed", "half")
    }
    means = {key[5:]: float(np.mean([r[key] for r in rows])) for key in rows[0] if key.startswith("loss_")}
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "smoke": manifest["smoke"],
        "training_rows": len(train_rows),
        "validation_rows": len(rows),
        "validation_episodes": len({r["episode_index"] for r in rows}),
        "trainable_parameters": sum(p.numel() for p in model.scheme4_utility_router.parameters()),
        "constants": constants,
        "action_mse": means,
        "probability": {
            "min": min(r["probability"] for r in rows),
            "max": max(r["probability"] for r in rows),
            "mean": float(np.mean([r["probability"] for r in rows])),
        },
        "paired_episode_bootstrap": paired,
        "affine_soft_mse_mean_abs_error": float(np.mean([abs(r["affine_soft_mse"] - r["loss_soft"]) for r in rows])),
        "passed_utility_gate": bool(
            not manifest["smoke"] and paired["train_optimal"]["ci95"][0] > 0 and means["soft"] < means["fixed"]
        ),
        "frozen_parameter_versions_unchanged": True,
        "strict_router_reload_passed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hashes": {
            name: sha256(output / name)
            for name in ("manifest.json", "train_rows.jsonl", "validation_rows.jsonl", "router.pt", "training.json")
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--prior", type=Path, default=Path("artifacts/adaptive_lookahead/20261110_utility_diagnostic_v3")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
