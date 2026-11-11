# Spatially Grounded Latent Lookahead (SG-LL)

SG-LL extends the PyTorch pi0.5 implementation with VGGT-supervised spatial
features, a current-conditioned future latent, and a future-latent residual
in the action expert. VGGT and future observations are training-only inputs.

This is a source release. Model weights, datasets, normalization statistics,
feature caches and raw experiment logs are external assets. A fresh clone
does not contain a runnable trained policy until those assets are prepared.

## Implementation

| Component | Entry point |
|---|---|
| Original pi0.5 | `src/openpi/models_pytorch/pi0_pytorch.py` |
| Spatial model / teacher extraction | `src/openpi/models_pytorch/pi0_align_pytorch.py`, `geometry_teacher.py` |
| SG-LL / historical Route A | `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py` (`Scheme4LACDoubleAlignPI0Pytorch`) |
| Stage-I latent tokenizer | `src/openpi/models_pytorch/delta_tokenizer.py` |
| Experimental utility router | `src/openpi/models_pytorch/scheme4_adaptive_lookahead.py` |
| Training and target alignment | `scripts/train_pi05_libero_proposals_pytorch.py`, `scripts/train_align_pytorch.py`, `src/openpi/training/data_loader.py` |
| Configuration | `src/openpi/training/config.py` |
| Serving | `scripts/serve_policy_proposal.py` |
| LIBERO evaluation | `examples/libero/run_parallel_eval.py` |
| Offline geometry cache / tokenizer training | `scripts/build_delta_vggt_cache.py`, `scripts/train_delta_tokenizer_from_cache.py` |
| Paired episode statistics | `scripts/aaai27/pi/paired_statistics.py` |

Historical Scheme/Route names remain in code for checkpoint compatibility.
Some earlier variants share imports with SG-LL and are retained; their
presence does not mean they are selected research directions.

## Environment

Run commands from this `SG-LL/` directory. Python 3.11, `uv`, and an NVIDIA
CUDA environment are required for training and policy evaluation. The supplied
lock describes the existing environment; a fresh installation has not been
revalidated as part of this source synchronization.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

The inherited pi0.5 implementation requires the supplied Transformers overlay
and Transformers 4.53.2. In a **new** environment, install the overlay into that
environment before constructing the model:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

Use the LIBERO-Plus package corresponding to the evaluation manifest; configure
its assets and task classification file according to its own instructions.
The evaluator accepts `--libero-pythonpath` explicitly. Its release default
is `../LIBERO-plus`; the package's installation and asset layout must still be
verified. Standard LIBERO and LIBERO-Plus episode sets are different protocols.

## Required assets

- Official pi0.5 PyTorch initialization, or a compatible SG-LL checkpoint.
  The optional JAX conversion entry is `examples/convert_jax_model_to_pytorch.py`.
- VGGT weights for training, placed in `checkpoints/vggt/` or supplied through
  `--vggt-weight-path`.
- The trained Stage-I h10 DeltaTokenizer for the selected SG-LL configuration;
  supply `--scheme4-tokenizer-checkpoint` when training, or
  `--policy-scheme4-tokenizer-checkpoint` when serving if relocation is needed.
- The selected LeRobot dataset, matching normalization statistics, and
  simulator assets for training/evaluation. LIBERO-Plus configs reference
  `Sylvest/libero_plus_lerobot`; set `HF_LEROBOT_HOME` to your data root.

Asset identifiers and historical protocols are described in the
[architecture audit](docs/SGLL_ARCHITECTURE_AUDIT.md) and
[execution notes](docs/scheme4_lac_double_align_execution.md). Those notes are
historical: replace path placeholders and explicitly match the checkpoint's
training configuration. The registered base config alone is **not** the full
published SG-LL training recipe; set horizon, losses, frozen parameters,
initialization, batch/accumulation and seed explicitly.

## Entry points

Inspect the actual CLI before specifying an experiment:

```bash
PYTHONPATH=.:src:packages/openpi-client/src \
  uv run --no-sync scripts/train_pi05_libero_proposals_pytorch.py \
  --proposal-variant scheme4_lac_double_align pi05_libero_plus_geo --help

PYTHONPATH=.:src:packages/openpi-client/src \
  uv run --no-sync scripts/serve_policy_proposal.py --help
```

Serving a prepared fixed-h10 policy, with a valid checkpoint and tokenizer:

```bash
PYTHONPATH=.:src:packages/openpi-client/src \
  uv run --no-sync scripts/serve_policy_proposal.py \
  --proposal-variant scheme4_lac_double_align \
  --policy-scheme4-lookahead-mode fixed \
  --policy-data-future-teacher-obs-offset 10 \
  --policy-scheme3-horizon-id 10 \
  --policy-scheme4-condition-negative-modes language,action \
  --policy-scheme4-tokenizer-checkpoint /path/to/h10_tokenizer/checkpoint.pt \
  policy:checkpoint --policy.config pi05_libero_plus_geo \
  --policy.dir /path/to/sgll_checkpoint
```

Keep the serving configuration consistent with the chosen checkpoint.
`none`, `fixed`, `soft`, and `hard` modes refer to the existing **single**
configured horizon. `soft`/`hard` require a compatible router checkpoint.
`none` retains the spatial path and disables future lookahead. It differs
from a separately trained delta-disabled model.

## CPU regression tests

These tests cover residual identity, gradients, intent selection, routing,
tokenizer round trips/overfitting, episode boundaries and serving overrides.
The command excludes the upstream ALOHA test that needs an external dataset.
These checks do not replace checkpoint loading or closed-loop GPU evaluation.

```bash
PYTHONPATH=.:src:packages/openpi-client/src OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  uv run --no-sync python -m pytest -q -k 'not test_with_real_dataset' \
  src/openpi/models_pytorch/delta_tokenizer_test.py \
  src/openpi/models_pytorch/scheme4_lac_double_align_test.py \
  src/openpi/models_pytorch/scheme4_adaptive_lookahead_test.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/training/external_benchmark_config_test.py \
  scripts/train_align_pytorch_partial_expert_test.py \
  scripts/serve_policy_proposal_test.py \
  scripts/aaai27/pi/paired_statistics_test.py \
  scripts/aaai27/pi/validate_hypothesis_c_scale_endpoint_test.py
```

## Research status and scope

- The fixed SG-LL and separately trained delta-disabled baselines have audited
  historical results. Their differences must be interpreted with the training
  and evaluation controls described in [results](docs/EXPERIMENT_RESULTS.md).
- The single-horizon utility-router prototype and tests exist, but its tested
  one-step action-loss objective did not improve on fixed lookahead.
- Fixed-residual cross-training-seed replication is ongoing. No final
  three-seed conclusion is included in this snapshot.
- Multi-horizon routing, VGGT-Ω and world-model distillation are proposed work,
  not implemented or validated claims in this release.

See the [improvement plan](docs/COUPLED_SPATIAL_WORLD_LOOKAHEAD_PLAN.md),
[experiment matrix](docs/EXPERIMENT_MATRIX.md), and
[implementation log](docs/IMPLEMENTATION_LOG.md). References to excluded raw
artifacts in these documents describe local evidence, not bundled downloads.

## Source synchronization

This copy changes only machine-specific default paths and prose, documented in
[SOURCE_RELEASE.md](docs/SOURCE_RELEASE.md). The active experiment checkout is
unchanged. Existing parent-repository history and unrelated code are excluded.

The inherited license is in [LICENSE](LICENSE). SG-LL builds on
[OpenPI](https://github.com/Physical-Intelligence/openpi),
[Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing), and
[VGGT](https://github.com/facebookresearch/vggt). Retained upstream notices
and separately downloaded assets retain their respective license terms.
