# SG-LL: Spatially Grounded Latent Lookahead

SG-LL is a vision-language-action (VLA) research project built on π0.5. It uses
VGGT supervision to learn a spatial representation and predicts a latent
representation of a future observation from the current policy context. The
predicted latent is fused into the action expert during flow-matching action
generation. The teacher and future observations are used for training; they are
not required by the fixed-lookahead policy at inference.

The repository also contains research code for selectively using the existing
fixed-horizon prediction. Adaptive multi-horizon lookahead and world-model
distillation are research directions, not established results in this release.

## Results and scope

The following historical comparison uses the LIBERO-Plus simulator and 10,030
matched episode keys across four suites. The aggregate is episode-weighted;
`LIBERO-10` is the long-horizon suite. The two rows are separately trained
models, not two inference settings of the same checkpoint.

| Model | Spatial | Object | Goal | LIBERO-10 | Aggregate |
|---|---:|---:|---:|---:|---:|
| Trained delta-disabled control | 92.21% | 90.39% | 81.63% | 83.17% | 86.75% |
| Fixed SG-LL, Δ=10 | 93.88% | 90.31% | 80.93% | 80.23% | 86.21% |

These are historical single-training-seed results. The control differs in
trainable parameters, so the comparison does not isolate the causal effect of
future fusion. The aggregate paired 95% bootstrap confidence interval for the
difference is [-1.157, +0.080] percentage points. See the
[evaluation record](docs/shared/2026-07-23_delta_disabled_four_suite_result.md)
for counts and statistical details. Multi-seed and optimization-aligned
comparisons remain ongoing.

## Setup

The following instructions target Linux with an NVIDIA GPU. The repository
specifies Python 3.11 in `.python-version`, PyTorch 2.11.0 with CUDA 12.8 wheels,
and Transformers 4.53.2 in [pyproject.toml](pyproject.toml). A compatible NVIDIA
driver, [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/)
are required. The lockfile is the source of dependency versions.

From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -e src/vggt
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

The final command replaces files **inside this project's virtual environment**;
the π0.5 PyTorch implementation checks for these changes. Use a dedicated
environment, and reapply the replacement if Transformers is reinstalled. The
commands describe the repository configuration; a clean installation has not
been independently verified for every driver or GPU. An existing compatible
π0.5 environment can also be used after checking the dependencies and patch.

## Assets

Model weights, datasets, caches, training artifacts, and results are not
included in the code repository. Prepare the following before training:

| Asset | Purpose and location |
|---|---|
| π0.5 LIBERO checkpoint | Initialize training from the official `pi05_libero` checkpoint, available at `gs://openpi-assets/checkpoints/pi05_libero/`; the PyTorch trainer expects a converted directory containing `model.safetensors` and matching normalization assets. See [checkpoint conversion](#checkpoint-conversion). |
| VGGT | Download [VGGT-1B `model.pt`](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) into a directory and pass that directory as `--vggt-weight-path`. |
| LIBERO-Plus training data | The `pi05_libero_plus_geo` config uses [`Sylvest/libero_plus_lerobot`](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot). Set `HF_LEROBOT_HOME` to its LeRobot cache root. |
| Δ=10 DeltaTokenizer | The historical fixed SG-LL training run requires its separately trained `checkpoint.pt`. A public download location is not currently specified in this repository. The relevant cache and tokenizer training entry points are [build_delta_vggt_cache.py](scripts/build_delta_vggt_cache.py) and [train_delta_tokenizer_from_cache.py](scripts/train_delta_tokenizer_from_cache.py); exact cache, checkpoint, and configuration provenance are needed for reproduction. |
| Normalization statistics | Use the statistics matching the training dataset and checkpoint. For this config the loader looks under `assets/pi05_libero_plus_geo/Sylvest/libero_plus_lerobot/`. Generate them with `uv run --no-sync python scripts/compute_norm_stats.py --config-name pi05_libero_plus_geo` if appropriate. Preserve and report the actual statistics used in any comparison. |
| LIBERO-Plus simulator | Evaluation additionally requires a working LIBERO-Plus installation, its assets and task classification, plus MuJoCo/EGL. The standard LIBERO example is a different evaluation setup. |

The SG-LL policy checkpoints and complete experiment manifests needed to
independently reproduce the historical table are not supplied by this README.
Without the exact tokenizer, normalization statistics, evaluation manifest,
and trained checkpoints, the commands below are useful for integration checks
but do not reproduce the reported numbers.

### Checkpoint conversion

The official checkpoint is in JAX format. After obtaining it, convert it into a
**new** output directory:

```bash
uv run --no-sync python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/pi05_libero \
  --config_name pi05_libero \
  --output_path /path/to/pi05_libero_torch
```

The converter copies the source checkpoint's `assets` directory when present;
it replaces an existing output `assets` directory, so use a fresh destination.
See [the conversion script](examples/convert_jax_model_to_pytorch.py) and the
[upstream LIBERO example](examples/libero/README.md) for details.

## Training smoke test

Set these paths to the assets prepared above. The dataset should be available
under `$HF_LEROBOT_HOME/Sylvest/libero_plus_lerobot`. Use a new experiment name
to avoid overwriting an existing run.

```bash
export HF_LEROBOT_HOME=/path/to/lerobot-cache
export PI05_CKPT=/path/to/pi05_libero_torch
export VGGT_DIR=/path/to/vggt
export TOKENIZER_CKPT=/path/to/h10-delta-tokenizer/checkpoint.pt
export TORCHDYNAMO_DISABLE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/train_pi05_libero_proposals_pytorch.py \
  --proposal-variant scheme4_lac_double_align pi05_libero_plus_geo \
  --exp-name sgll_h10_smoke --no-wandb-enabled \
  --seed 42 --num-train-steps 1 --batch-size 1 \
  --gradient-accumulation-steps 1 --num-workers 0 \
  --pytorch-weight-path "$PI05_CKPT" --vggt-weight-path "$VGGT_DIR" \
  --no-train-action-expert --current-geo-loss-coeff 0.06 --delta-loss-coeff 0.0 \
  --scheme2-injection-mode residual_prefix \
  --scheme4-tokenizer-checkpoint "$TOKENIZER_CKPT" \
  --scheme4-horizon-id 10 --scheme4-delta-token-loss-coeff 0.0 \
  --scheme4-future-recon-loss-coeff 0.01 \
  --data.future-teacher-obs-offset 10 --log-interval 1 --save-interval 1
```

The command performs one update only; it is **not** the reported training run.
The config's default future observation offset is 20, so both the tokenizer
horizon and the data offset are explicitly set to 10 here. The historical run
used 10,000 steps, seed 42, batch size 16, frozen VLM and action expert, and
the loss settings above. Its recorded command and assumptions are in
[the execution notes](docs/scheme4_lac_double_align_execution.md). Historical
launch scripts contain machine-specific paths and may overwrite outputs;
review them before use.

## LIBERO-Plus evaluation smoke test

Install LIBERO-Plus and its simulation dependencies separately, then set the
following paths. Point `POLICY_DIR` to a saved SG-LL checkpoint directory
containing `model.safetensors` and its matching assets. The example evaluates
one task with one trial; it is a wiring check, not a benchmark result.

```bash
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export POLICY_DIR=/path/to/sgll/checkpoint
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run --no-sync python examples/libero/run_parallel_eval.py \
  --run-name sgll_h10_smoke_eval \
  --task-suite-name libero_spatial --task-start 0 --num-tasks 1 \
  --num-workers 1 --num-trials-per-task 1 --seed 7 --port 8240 --gpu 0 \
  --policy-config pi05_libero_plus_geo --policy-dir "$POLICY_DIR" \
  --proposal-variant scheme4_lac_double_align \
  --policy-data-future-teacher-obs-offset 10 \
  --libero-pythonpath "$LIBERO_PLUS_ROOT" \
  --mujoco-egl-device-id 0 --server-ready-timeout-sec 900 --no-save-videos
```

`--policy-scheme4-lookahead-mode` accepts `none`, `fixed`, `hard`, and `soft`.
The adaptive options route the existing fixed-horizon branch; they should not
be interpreted as a validated multi-horizon policy. Compare models using the
same task classification, episode manifest, environment seeds, preprocessing,
and evaluation protocol, and report training seeds separately.

## Code and documentation

- [SG-LL model variants](src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py)
  and [π0.5 spatial model](src/openpi/models_pytorch/pi0_align_pytorch.py).
- [Training configurations](src/openpi/training/config.py) and
  [training entry point](scripts/train_pi05_libero_proposals_pytorch.py).
- [Architecture audit](docs/SGLL_ARCHITECTURE_AUDIT.md),
  [experiment matrix](docs/EXPERIMENT_MATRIX.md), and
  [experiment results](docs/EXPERIMENT_RESULTS.md).
- [Real-robot ALOHA example](examples/aloha_real/README.md).

## Acknowledgements and license

This project builds on [openpi](https://github.com/Physical-Intelligence/openpi),
[VGGT](https://github.com/facebookresearch/vggt), and the LIBERO family of
benchmarks. The repository is distributed under the [Apache 2.0 license](LICENSE);
third-party code, models, datasets, and benchmarks may have separate licenses.

