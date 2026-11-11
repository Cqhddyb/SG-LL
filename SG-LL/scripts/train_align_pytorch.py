"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

from collections.abc import Sequence
import contextlib
import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
from openpi.models_pytorch import pi0_align_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from vggt.models.vggt import VGGT


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def unwrap_ddp(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _set_requires_grad(module: torch.nn.Module, *, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


@dataclasses.dataclass(frozen=True)
class TrainableParameterGroup:
    """Optimizer-group contract after applying the explicit freeze policy."""

    name: str
    params: list[torch.nn.Parameter]
    lr_scale: float


_PARTIAL_EXPERT_AUX_MODULES = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


def _validate_partial_action_expert_config(config: _config.TrainConfig) -> tuple[int, float | None]:
    """Validate the deliberately narrow partial action-expert ablation."""

    last_n_blocks = int(getattr(config, "partial_action_expert_last_n_blocks", 0))
    expert_lr = getattr(config, "partial_action_expert_lr", None)
    if last_n_blocks < 0:
        raise ValueError(f"partial_action_expert_last_n_blocks must be >= 0, got {last_n_blocks}")
    if last_n_blocks == 0:
        if expert_lr is not None:
            raise ValueError("partial_action_expert_lr requires partial_action_expert_last_n_blocks > 0")
        return 0, None
    if not getattr(config, "freeze_vlm_backbone", True):
        raise ValueError("partial action-expert adaptation requires freeze_vlm_backbone=True")
    if getattr(config, "train_action_expert", True):
        raise ValueError("partial action-expert adaptation cannot combine with train_action_expert=True")
    if not getattr(config, "freeze_vggt", True):
        raise ValueError("partial action-expert adaptation requires freeze_vggt=True")
    if expert_lr is None or float(expert_lr) <= 0.0:
        raise ValueError("partial action-expert adaptation requires partial_action_expert_lr > 0")
    return last_n_blocks, float(expert_lr)


def _action_expert_layers(model_to_train: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Return the actual Gemma action-expert block sequence with no name guesses."""

    try:
        layers = model_to_train.paligemma_with_expert.gemma_expert.model.layers
    except AttributeError as error:
        raise RuntimeError("model has no Gemma action-expert Transformer layer sequence") from error
    if not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)):
        raise RuntimeError(f"Gemma action-expert layers must be ModuleList/Sequential, got {type(layers).__name__}")
    if not layers:
        raise RuntimeError("Gemma action-expert layer sequence is empty")
    return layers


def _freeze_action_expert_and_auxiliary_modules(model_to_train: torch.nn.Module) -> None:
    _set_requires_grad(model_to_train.paligemma_with_expert.gemma_expert, requires_grad=False)
    for module_name in _PARTIAL_EXPERT_AUX_MODULES:
        module = getattr(model_to_train, module_name, None)
        if module is not None:
            _set_requires_grad(module, requires_grad=False)


def _partial_expert_parameter_names(
    model_to_train: torch.nn.Module,
    selected_layers: Sequence[torch.nn.Module],
) -> set[str]:
    selected_param_ids = {id(param) for layer in selected_layers for param in layer.parameters()}
    if not selected_param_ids:
        raise RuntimeError("selected action-expert blocks have no parameters")
    selected_names = {name for name, param in model_to_train.named_parameters() if id(param) in selected_param_ids}
    if len(selected_names) != len(selected_param_ids):
        raise RuntimeError("could not map every selected action-expert parameter to exactly one model name")
    expected_prefixes = tuple(
        f"paligemma_with_expert.gemma_expert.model.layers.{layer_index}."
        for layer_index in range(
            len(_action_expert_layers(model_to_train)) - len(selected_layers),
            len(_action_expert_layers(model_to_train)),
        )
    )
    unexpected_names = sorted(name for name in selected_names if not name.startswith(expected_prefixes))
    if unexpected_names:
        raise RuntimeError(
            f"selected parameters are not confined to the final action-expert blocks: {unexpected_names}"
        )
    return selected_names


def _validate_partial_expert_freeze_boundary(model_to_train: torch.nn.Module, selected_names: set[str]) -> None:
    expert_prefix = "paligemma_with_expert.gemma_expert."
    unexpected_trainable = sorted(
        name
        for name, param in model_to_train.named_parameters()
        if name.startswith(expert_prefix) and param.requires_grad and name not in selected_names
    )
    if unexpected_trainable:
        raise RuntimeError(
            f"partial action-expert adaptation left non-tail expert parameters trainable: {unexpected_trainable}"
        )


def build_optimizer_parameter_groups(
    groups: Sequence[TrainableParameterGroup], *, peak_lr: float
) -> list[dict[str, object]]:
    """Build AdamW groups while retaining scheduler-stable relative LR scales."""

    if peak_lr <= 0.0:
        raise ValueError(f"peak_lr must be positive, got {peak_lr}")
    seen_param_ids: set[int] = set()
    optimizer_groups: list[dict[str, object]] = []
    for group in groups:
        if not group.params:
            raise ValueError(f"optimizer group {group.name!r} is empty")
        if group.lr_scale <= 0.0:
            raise ValueError(f"optimizer group {group.name!r} must have positive lr_scale")
        param_ids = {id(param) for param in group.params}
        if len(param_ids) != len(group.params):
            raise ValueError(f"optimizer group {group.name!r} contains duplicate parameters")
        overlap = seen_param_ids.intersection(param_ids)
        if overlap:
            raise ValueError(f"optimizer groups overlap on {len(overlap)} parameters")
        seen_param_ids.update(param_ids)
        optimizer_groups.append(
            {
                "params": group.params,
                "lr": peak_lr * group.lr_scale,
                "lr_scale": group.lr_scale,
                "group_name": group.name,
            }
        )
    return optimizer_groups


def set_scheduled_learning_rates(optimizer: torch.optim.Optimizer, scheduled_lr: float) -> None:
    """Apply the planning schedule without collapsing explicit group ratios."""

    if scheduled_lr <= 0.0:
        raise ValueError(f"scheduled_lr must be positive, got {scheduled_lr}")
    for parameter_group in optimizer.param_groups:
        lr_scale = float(parameter_group.get("lr_scale", 1.0))
        parameter_group["lr"] = scheduled_lr * lr_scale


def configure_trainable_parameters(
    model, vggt_model, config: _config.TrainConfig, *, is_main: bool
) -> list[TrainableParameterGroup]:
    """Apply freezing and return non-overlapping, scheduler-stable trainable groups.

    With ``partial_action_expert_last_n_blocks=0`` this preserves the prior
    one-group behavior. Partial mode freezes the complete action expert first,
    then re-enables only the final N Gemma Transformer blocks.
    """
    model_to_train = unwrap_ddp(model)
    partial_last_n_blocks, partial_expert_lr = _validate_partial_action_expert_config(config)

    if getattr(config, "scheme4_router_only_training", False):
        if not getattr(config, "scheme4_adaptive_lookahead_enabled", False):
            raise ValueError("scheme4_router_only_training requires scheme4_adaptive_lookahead_enabled=True")
        if not getattr(model_to_train, "scheme4_router_only_training", False):
            raise ValueError("router-only training config requires a Scheme4 model built with the same option")
        if partial_last_n_blocks:
            raise ValueError("scheme4_router_only_training cannot combine with partial action-expert training")
        router = getattr(model_to_train, "scheme4_utility_router", None)
        if router is None:
            raise ValueError("scheme4_router_only_training requires a constructed Scheme4 utility router")

        _set_requires_grad(model_to_train, requires_grad=False)
        _set_requires_grad(router, requires_grad=True)
        _set_requires_grad(vggt_model, requires_grad=False)
        named_trainable = [(name, param) for name, param in model_to_train.named_parameters() if param.requires_grad]
        if not named_trainable or any(not name.startswith("scheme4_utility_router.") for name, _ in named_trainable):
            raise RuntimeError("router-only freeze boundary exposed a non-router trainable parameter")
        trainable_params = [param for _, param in named_trainable]
        if is_main:
            trainable_count = sum(param.numel() for param in trainable_params)
            total_count = sum(param.numel() for param in model_to_train.parameters())
            logging.info(
                "Router-only parameters: %.2fM / %.2fM; all VLA, spatial, future and VGGT weights frozen",
                trainable_count / 1e6,
                total_count / 1e6,
            )
        return [TrainableParameterGroup(name="lookahead_router", params=trainable_params, lr_scale=1.0)]

    if getattr(config, "freeze_vggt", True):
        _set_requires_grad(vggt_model, requires_grad=False)

    if getattr(config, "freeze_vlm_backbone", True):
        _set_requires_grad(model_to_train.paligemma_with_expert.paligemma, requires_grad=False)

    if not getattr(config, "train_action_expert", True):
        _freeze_action_expert_and_auxiliary_modules(model_to_train)

    partial_expert_names: set[str] = set()
    if partial_last_n_blocks:
        action_expert_layers = _action_expert_layers(model_to_train)
        if partial_last_n_blocks > len(action_expert_layers):
            raise ValueError(
                "partial_action_expert_last_n_blocks="
                f"{partial_last_n_blocks} exceeds action-expert depth {len(action_expert_layers)}"
            )
        selected_layers = list(action_expert_layers[-partial_last_n_blocks:])
        for layer in selected_layers:
            _set_requires_grad(layer, requires_grad=True)
        partial_expert_names = _partial_expert_parameter_names(model_to_train, selected_layers)
        _validate_partial_expert_freeze_boundary(model_to_train, partial_expert_names)

    named_trainable = [(name, param) for name, param in model_to_train.named_parameters() if param.requires_grad]
    trainable_params = [param for _, param in named_trainable]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    if is_main:
        logging.info(
            "Trainable parameters: %.2fM / %.2fM (freeze_vlm_backbone=%s, train_action_expert=%s, partial_expert_blocks=%s)",
            trainable_count / 1e6,
            total_params / 1e6,
            getattr(config, "freeze_vlm_backbone", True),
            getattr(config, "train_action_expert", True),
            partial_last_n_blocks,
        )
    if not trainable_params:
        raise ValueError("No trainable parameters left after applying freeze settings.")

    if not partial_last_n_blocks:
        return [TrainableParameterGroup(name="default", params=trainable_params, lr_scale=1.0)]

    planning_params = [param for name, param in named_trainable if name not in partial_expert_names]
    expert_tail_params = [param for name, param in named_trainable if name in partial_expert_names]
    peak_lr = float(config.lr_schedule.peak_lr)
    assert partial_expert_lr is not None
    expert_lr_scale = partial_expert_lr / peak_lr
    groups = [
        TrainableParameterGroup(name="planning", params=planning_params, lr_scale=1.0),
        TrainableParameterGroup(name="action_expert_tail", params=expert_tail_params, lr_scale=expert_lr_scale),
    ]
    build_optimizer_parameter_groups(groups, peak_lr=peak_lr)
    if is_main:
        logging.info(
            "Partial action expert: final %s/%s blocks, planning_lr=%.2e, expert_tail_lr=%.2e (scale=%.3f)",
            partial_last_n_blocks,
            len(_action_expert_layers(model_to_train)),
            peak_lr,
            partial_expert_lr,
            expert_lr_scale,
        )
    return groups


def scalarize_metrics(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.detach().float().mean().cpu())
    return metrics


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(
            f"batch_size={config.batch_size} must be divisible by world_size={world_size} for DDP training."
        )
    gradient_accumulation_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}.")

    micro_batch_size_per_gpu = config.batch_size // world_size
    effective_train_batch_size = config.batch_size * gradient_accumulation_steps
    logging.info(
        "Using micro-batch size per GPU: %s (global micro-batch: %s, accumulation: %s, effective train batch: %s)",
        micro_batch_size_per_gpu,
        config.batch_size,
        gradient_accumulation_steps,
        effective_train_batch_size,
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = pi0_align_pytorch.PI0Pytorch(model_cfg, config).to(device)
    vggt_model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    ).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch weight file not found at {model_path}")
        safetensors.torch.load_model(
            model,
            model_path,
            strict=False,
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    if config.vggt_weight_path is not None:
        vggt_path = os.path.join(config.vggt_weight_path, "model.pt")
        if not os.path.exists(vggt_path):
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_path}")
        vggt_model.load_state_dict(torch.load(vggt_path), strict=False)
        logging.info(f"Loaded VGGT weights from {config.vggt_weight_path}")

    trainable_parameter_groups = configure_trainable_parameters(model, vggt_model, config, is_main=is_main)
    trainable_params = [param for group in trainable_parameter_groups for param in group.params]

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optimizer_parameter_groups = build_optimizer_parameter_groups(
        trainable_parameter_groups,
        peak_lr=peak_lr,
    )
    optim = torch.optim.AdamW(
        optimizer_parameter_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    if getattr(config, "scheme4_router_only_training", False):
        model.eval()
        model_to_train = unwrap_ddp(model)
        model_to_train.scheme4_utility_router.train()
    vggt_model.eval()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            "Training config: batch_size=%s, micro_batch_size_per_gpu=%s, gradient_accumulation_steps=%s, "
            "effective_train_batch_size=%s, num_optimizer_steps=%s",
            config.batch_size,
            micro_batch_size_per_gpu,
            gradient_accumulation_steps,
            effective_train_batch_size,
            config.num_train_steps,
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    loader_iter = iter(loader)
    loader_epoch = 0
    optim.zero_grad(set_to_none=True)

    while global_step < config.num_train_steps:
        # Update LR once per optimizer step.
        set_scheduled_learning_rates(optim, lr_schedule(global_step))

        micro_infos = []
        last_loss_value = float("nan")

        for accumulation_idx in range(gradient_accumulation_steps):
            try:
                observation, actions = next(loader_iter)
            except StopIteration:
                loader_epoch += 1
                if use_ddp and hasattr(loader, "set_epoch"):
                    loader.set_epoch(loader_epoch)
                loader_iter = iter(loader)
                observation, actions = next(loader_iter)

            # The unified data loader returns (observation, actions) tuple.
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            sync_context = (
                model.no_sync()
                if use_ddp and gradient_accumulation_steps > 1 and accumulation_idx < gradient_accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                outputs = model(observation, actions, vggt=vggt_model)
                loss = outputs["loss"]
                last_loss_value = float(loss.detach().cpu())
                (loss / gradient_accumulation_steps).backward()

            if is_main:
                micro_infos.append(scalarize_metrics(outputs))

        # Log memory usage after the full accumulated backward pass.
        if global_step < 5 and is_main and torch.cuda.is_available():
            log_memory_usage(device, global_step, f"after_backward_accum_{gradient_accumulation_steps}")

        # Gradient clipping sees the accumulated gradient.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=config.optimizer.clip_gradient_norm,
        )

        optim.step()
        optim.zero_grad(set_to_none=True)

        # Clear gradients more aggressively.
        for param in trainable_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

        if is_main:
            step_info = {}
            metric_keys = sorted({key for info in micro_infos for key in info})
            for key in metric_keys:
                values = [info[key] for info in micro_infos if key in info]
                step_info[key] = sum(values) / max(1, len(values))
            step_info["learning_rate"] = optim.param_groups[0]["lr"]
            for parameter_group in optim.param_groups:
                group_name = str(parameter_group.get("group_name", "default"))
                step_info[f"learning_rate_{group_name}"] = float(parameter_group["lr"])
            step_info["grad_norm"] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            step_info["gradient_accumulation_steps"] = float(gradient_accumulation_steps)
            step_info["effective_train_batch_size"] = float(effective_train_batch_size)
            infos.append(step_info)

        if is_main and (global_step % config.log_interval == 0):
            elapsed = time.time() - start_time

            # Average stats over log interval, one entry per optimizer step.
            metric_keys = sorted({key for info in infos for key in info})
            avg_metrics = {
                key: sum(info[key] for info in infos if key in info) / max(1, sum(key in info for info in infos))
                for key in metric_keys
            }
            avg_lr = avg_metrics.get("learning_rate", optim.param_groups[0]["lr"])

            avg_grad_norm = avg_metrics.get("grad_norm")
            loss_parts = " ".join(
                f"{key}={avg_metrics[key]:.4f}"
                for key in (
                    "loss",
                    "flow_loss",
                    "base_flow_loss",
                    "current_geo_loss",
                    "delta_geo_loss",
                    "delta_main_loss",
                    "delta_norm_loss",
                    "delta_global_cos_loss",
                    "delta_valid_ratio",
                    "teacher_delta_norm_valid",
                    "pred_delta_norm_valid",
                    "residual_abs_mean",
                    "residual_reg",
                    "spatial_token_cos_loss",
                    "spatial_value_loss",
                    "spatial_relation_loss",
                    "scheme1_refined_visual_norm",
                    "scheme1_rq_query_norm",
                    "scheme1_rq_gate_abs_mean",
                    "scheme2_aux_abs_mean",
                    "scheme2_aux_tokens",
                    "scheme2_gate_abs_mean",
                    "scheme2_gate_max_abs",
                    "scheme2_injection_is_concat",
                    "scheme3_d3_delta_abs_mean",
                    "scheme3_d3_action_hat_abs_mean",
                    "scheme3_d3_gate_abs_mean",
                    "scheme3_d3_gate_max_abs",
                    "scheme3_delta_token_loss",
                    "scheme3_delta_future_loss",
                    "scheme3_delta_current_as_future_loss",
                    "scheme3_delta_oracle_future_loss",
                    "scheme3_delta_abs_mean",
                    "scheme3_delta_gate_abs_mean",
                    "scheme3_delta_gate_max_abs",
                    "scheme3_delta_token_smooth_l1",
                    "scheme3_delta_token_cosine",
                    "scheme3_future_geometry_value_loss",
                    "scheme3_future_geometry_cosine_loss",
                    "scheme3_future_geometry_relation_loss",
                    "scheme3_delta_future_fallback_ratio",
                    "future_recon_loss",
                    "delta_token_loss",
                    "spatial_gate_abs_mean",
                    "spatial_gate_max_abs",
                    "delta_gate_abs_mean",
                    "delta_gate_max_abs",
                    "delta_token_norm",
                    "delta_residual_norm",
                    "language_summary_norm",
                    "action_intent_norm",
                    "action_condition_norm",
                    "language_token_residual_norm",
                    "scheme4_delta_enabled",
                    "scheme4_spatial_norm",
                    "scheme4_aux_abs_mean",
                    "scheme4_current_as_future_loss",
                    "scheme4_oracle_future_loss",
                    "scheme4_future_fallback_ratio",
                    "scheme4_delta_token_smooth_l1",
                    "scheme4_delta_token_cosine",
                    "scheme4_future_geometry_value_loss",
                    "scheme4_future_geometry_cosine_loss",
                    "scheme4_future_geometry_relation_loss",
                    "scheme4_utility_loss",
                    "scheme4_utility_benefit_mean",
                    "scheme4_utility_target_mean",
                    "scheme4_router_probability_mean",
                    "scheme4_router_selected_fraction",
                    "scheme4_training_intent_t1",
                    "scheme4_current_geo_loss_coeff",
                    "scheme4_delta_token_loss_coeff",
                    "scheme4_future_recon_loss_coeff",
                    "scheme4_condition_negative_loss",
                    "scheme4_condition_negative_loss_coeff",
                    "scheme4_condition_negative_margin",
                    "scheme4_condition_normal_delta_distance",
                    "scheme4_condition_neg_language_loss",
                    "scheme4_condition_neg_language_delta_distance",
                    "scheme4_condition_neg_language_valid",
                    "scheme4_condition_neg_action_loss",
                    "scheme4_condition_neg_action_delta_distance",
                    "scheme4_condition_neg_action_valid",
                    "scheme4_grounded_gt_action_norm",
                    "scheme4_grounded_decoder_action_norm",
                    "scheme4_grounded_decoder_uses_student_intent",
                )
                if key in avg_metrics
            )
            logging.info(
                f"step={global_step} {loss_parts} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} "
                f"accum={gradient_accumulation_steps} eff_batch={effective_train_batch_size} time={elapsed:.1f}s"
                if avg_grad_norm is not None
                else f"step={global_step} {loss_parts} lr={avg_lr:.2e} "
                f"accum={gradient_accumulation_steps} eff_batch={effective_train_batch_size} time={elapsed:.1f}s"
            )

            # Log to wandb.
            if config.wandb_enabled and len(infos) > 0:
                log_payload = {
                    **avg_metrics,
                    "learning_rate": avg_lr,
                    "step": global_step,
                    "time_per_step": elapsed / config.log_interval,
                }
                wandb.log(log_payload, step=global_step)

            start_time = time.time()
            infos = []  # Reset stats collection

        global_step += 1
        # Save checkpoint using the new mechanism.
        save_checkpoint(model, optim, global_step, config, is_main, data_config)

        # Update progress bar, one tick per optimizer step.
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{last_loss_value:.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    "step": global_step,
                    "accum": gradient_accumulation_steps,
                }
            )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
