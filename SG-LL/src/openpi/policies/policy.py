from collections.abc import Sequence
import hashlib
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared.request_noise import NOISE_PROTOCOL
from openpi.shared.request_noise import NOISE_SEED_KEY

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _consume_pytorch_inference_diagnostics(model: object) -> dict[str, float | list[float]] | None:
    """Consume optional batch-one diagnostics without changing normal responses."""

    consumer = getattr(model, "take_scheme4_inference_diagnostics", None)
    if consumer is None:
        return None
    raw = consumer()
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("Scheme4 inference diagnostics must be a non-empty tensor dictionary")
    converted: dict[str, float | list[float]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("Scheme4 inference diagnostics must map string names to tensors")
        if value.ndim not in {1, 2} or value.shape[0] != 1:
            raise RuntimeError(f"Scheme4 diagnostic {name} must be [1] or [1,S], got {tuple(value.shape)}")
        array = value[0].detach().float().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"Scheme4 diagnostic {name} contains non-finite values")
        converted[name] = float(array) if array.ndim == 0 else array.tolist()
    return converted


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Reserved, opt-in evaluation metadata is removed before transforms.
        # Use a private generator: other clients cannot advance this stream.
        request_noise = None
        if NOISE_SEED_KEY in obs:
            obs = dict(obs)
            seed = obs.pop(NOISE_SEED_KEY)
            if not self._is_pytorch_model:
                raise ValueError("Request noise seeds currently require a PyTorch policy")
            if type(seed) is not int or not 0 <= seed < (1 << 63):
                raise ValueError("Request noise seed must be an integer in [0, 2**63)")
            if noise is not None or "noise" in self._sample_kwargs:
                raise ValueError("Request noise seed and explicit noise are mutually exclusive")
            generator = torch.Generator(device=self._pytorch_device).manual_seed(seed)
            # [H,A], matching the model's float32 standard normal action noise.
            sampled_noise = torch.randn(
                (self._model.config.action_horizon, self._model.config.action_dim),
                dtype=torch.float32,
                device=self._pytorch_device,
                generator=generator,
            )
            noise = sampled_noise.cpu().numpy()
            request_noise = {
                "protocol": NOISE_PROTOCOL,
                "seed": seed,
                "sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
            }
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampled_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        inference_diagnostics = _consume_pytorch_inference_diagnostics(self._model) if self._is_pytorch_model else None
        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if inference_diagnostics is not None:
            outputs["scheme4_inference_diagnostics"] = inference_diagnostics
        if request_noise is not None:
            outputs["action_noise"] = request_noise
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
