# SG-LL Architecture Audit

## Scope and evidence

This audit is based on the checked-out PyTorch implementation and the existing
experiment records. It separates facts confirmed by code from claims that still
need an experiment. The repository root for the implementation is
`SG-LL/`; the parent Git worktree contains unrelated pre-existing changes and
was not reset or cleaned.

## Confirmed execution path

1. `scripts/train_pi05_libero_proposals_pytorch.py` selects
   `Scheme4LACDoubleAlignPI0Pytorch` from
   `src/openpi/models_pytorch/pi0_proposal_variants_pytorch.py` and delegates
   optimization to `scripts/train_align_pytorch.py`.
2. `PI0Pytorch`/the π0.5 implementation embeds image and language prefixes in
   `src/openpi/models_pytorch/pi0_align_pytorch.py`, and embeds robot state,
   noisy action suffix and flow time through `embed_suffix`.
3. The frozen VLA prefix hidden state is passed to `scheme2_adapter`. Its
   spatial bottleneck is retained as the spatial prefix intervention. In
   Scheme4, the language summary is a masked mean over the language portion of
   the prefix hidden state. Legacy Route A action intent is a detached suffix
   mean: training uses random flow time and a noisy demonstrated-action suffix,
   whereas serving uses the initial pure-noise suffix at t=1. This mismatch is
   confirmed in code. An opt-in adaptive training mode now computes a detached
   intent pass at t=1 from the same Gaussian noise tensor used to construct the
   flow-matching training suffix; the default remains legacy-compatible. The
   new train/serve paths have not yet been shown numerically equivalent.
4. `FutureTeacherLeRobotDataset` in
   `src/openpi/training/data_loader.py` constructs a future image by adding
   `data.future_teacher_obs_offset` to the current dataset index. The target is
   clamped to the end of the current episode and emits
   `future_teacher_fallback=True` when clamping occurs. The current and future
   images are used only by the training-time VGGT teacher path.
5. VGGT is loaded by the alignment trainer and its features are extracted by
   `_encode_vggt_hidden` in `pi0_align_pytorch.py`. The teacher is detached;
   the student spatial adapter and Scheme4 predictor receive the losses.
6. The offset-10 DeltaTokenizer is loaded from
   `artifacts/delta_tokenizer_cached/s3_m4_h10_n2048_seed31_tok_m8_steps4000_lr3e4_noid/checkpoint.pt`.
   It encodes the current/future VGGT feature transition for the auxiliary
   future target and decodes a predicted future feature for reconstruction.
7. `Scheme4LanguageActionDeltaAdapter` predicts delta tokens from spatial
   tokens, masked language summary, robot state and detached action intent. Its
   residual is inserted into the action suffix before `action_out_proj`.
8. During flow matching, `sample_actions` computes the spatial prefix and one
   initial suffix intent pass, then reuses the prefix KV cache. Future tokens
   are reused while the suffix residual is recomputed in each denoising step
   by `denoise_step`; no VGGT,
   future frame or ground-truth action is run at inference.

## Trainability and gradient boundaries

The default Route A configuration freezes the VLM and action expert and trains
the spatial/future adapters. The VGGT and DeltaTokenizer are frozen. The
action-intent feature is detached, so future losses do not backpropagate into
the action expert through that condition. The trainer has explicit support for
freezing the action expert and for a partial tail-block ablation.

The adaptive router, when enabled, is a small trainable module. It receives
only policy-available features and predicts a probability for using the
already-configured single horizon. Its utility target is the detached proxy
`base_action_loss - lookahead_action_loss`. In the opt-in router-only mode, all
pre-existing tensors are frozen, only the router is trainable, and the flow
loss cannot update the router through the residual scale; router gradients come
from utility supervision alone. Both one-step and final1,000-step checkpoint
comparisons verified1,143 common tensors unchanged and22 new router tensors
(737,089 parameters). Evidence: pilot `freeze_verification.json` under
`artifacts/adaptive_lookahead/20261109_router_pilot/`.

The optional evaluation-only `_policy_noise_seed` request field is stripped
before transforms and uses a private Torch generator to feed the existing
explicit action-noise path. Without it, legacy RNG behavior is retained.
The two-suite fixed-mode repeat verifies exact actions/diagnostics across70
requests; common seeded request-noise hashes match across modes. This ensures
matched draws at matching request indices, not identical states after policy
trajectories diverge. Decision: `docs/decisions/2026-11-10_request_noise_pairing.md`.

## Existing empirical record

The authoritative paired four-suite record is
`docs/shared/2026-07-23_delta_disabled_four_suite_result.md`:

| Method | Spatial | Object | Goal | Long/LIBERO-10 | Aggregate |
|---|---:|---:|---:|---:|---:|
| Route A fixed SG-LL | 93.88% | 90.31% | 80.93% | 80.23% | 86.21% |
| Trained delta-disabled control | 92.21% | 90.39% | 81.63% | 83.17% | 86.75% |

The comparison is matched by episode keys and has no duplicate/error rows.
Route A is significantly better only on Spatial; the trained no-delta control
is significantly better on LIBERO-10. This rules out an aggregate superiority
claim for fixed SG-LL.

## Limitations that determine the next experiment

- `scheme4_inference_lookahead_mode` supports `none`, `fixed`, `soft` and
  `hard`, but the router currently selects only whether to use the configured
  horizon. It does not yet select among `{0,5,10,20,40}`.
- The legacy Route A configuration still uses sampled-flow-time action intent
  in training. The adaptive experiment explicitly opts into a detached `t=1`
  noise-suffix pass. This distributional mismatch is addressed for that arm,
  but exact train/serve activation equivalence has not yet been measured.
- The single h=10 tokenizer and h=10 future target do not support a valid
  multi-horizon claim by changing a configuration integer alone.
- The current control result is matched in data/steps but is not parameter
  identical to Route A. A parameter-matched router-only pilot is therefore the
  least ambiguous next causal test.

## Confirmed non-leakage properties

At serving time, future images, future states, VGGT and ground-truth actions
are absent. `none` explicitly keeps the spatial prefix while skipping the
auxiliary intent pass and future predictor, producing zero future residual.
This contract is covered at the fusion-module level and remains to be checked
as a full-model paired fixed-noise intervention.

