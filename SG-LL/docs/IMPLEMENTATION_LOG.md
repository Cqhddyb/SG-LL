# Implementation Log

## 2026-11-03 — audit and GPU gate

- Confirmed the repository root is the parent `Spatial-Forcing` Git worktree;
  the project implementation is in `SG-LL/`.
- Preserved all pre-existing parent-worktree deletions, modifications and
  untracked files. No reset, clean or destructive deletion was run.
- Confirmed π0.5, VGGT, h=10 DeltaTokenizer, LIBERO-Plus data and existing
  paired results are present.
- Added the architecture audit, coupled plan and experiment matrix.
- Restricted shell: no `/dev/nvidia*`, `torch.cuda.is_available()==False`.
- Escalated diagnostic: GPU 0 and GPU 1 are RTX 5090 with 32,607 MiB each;
  both were busy at diagnostic time. Training is not yet claimed or started.
- The prior handoff records 25 passing adaptive/Scheme4 tests. The separate
  loader/config run had 13 passes and one unrelated ALOHA cache failure;
  these were not serving-test results and were not rerun on November 6.

## 2026-11-06 — real GPU attempt, resource collision

- Both 5090 cards initially reported over 31 GB free. GPU-enabled `.venv`
  PyTorch 2.11.0+cu128 successfully ran a finite CUDA matrix multiplication.
- Reused the existing environment with explicit `PYTHONPATH=src:packages/openpi-client/src`;
  no package install or environment rebuild was performed.
- Ran the exact one-step fixed-h10 command in
  `artifacts/adaptive_lookahead/20261106_gpu_smoke/run.sh` on physical GPU 1,
  B1, seed42, zero workers, no W&B. The real LIBERO-Plus loader read existing
  normalization and generated its 2,238,036-row Arrow split in a writable cache.
- The attempt exited with CUDA OOM at `model.to(device)`, before loading policy
  weights, any training batch forward/backward, optimizer step, or checkpoint
  save. The competing PID 74505 (`airbot-pi0.5` training outside this project)
  occupied 29,892 / 29,888 MiB on GPUs 0 / 1 at the post-failure check.
- Preserved the failure log; removed only the empty experiment checkpoint
  directory created by this attempt. Retained the useful 214 MB dataset cache
  to avoid rebuilding it on the next attempt. No user files or processes touched.
- No model, loss, trainer, preprocessing, or evaluation source changed. Checked
  the launcher's Bash syntax and generated receipt; no new model tests claimed.

## 2026-11-09 — fixed and router-only smoke; contracts added

- Fixed h=10 one-step GPU smoke on RTX 5090 0 passed end to end. Exact command
  is `artifacts/adaptive_lookahead/20261109_gpu_smoke/run.sh`; seed42, B1,
  effective batch1, offset10, tokenizer h10. It loaded existing official
  π0.5 and VGGT weights, completed one real LIBERO-Plus batch, backward and
  AdamW update, and saved `checkpoints/pi05_libero_plus_geo/sgll_fixed_h10_gpu_smoke_20261109/1`.
  Loss `0.0862`, grad norm `0.13`, peak allocated `12.78 GB`, step time `18.32 s`.
- Added explicit `scheme4_training_intent_timestep=t1_noise` for training/serve
  alignment; default `sampled_flow` preserves existing checkpoints. In this
  mode a detached suffix pass uses the same initial noise and `t=1` as inference.
- Added opt-in `scheme4_router_only_training`: all model and VGGT weights are
  frozen except the utility router; model remains in eval mode, router in train
  mode, and flow-fusion gate is stop-gradient so only utility loss trains it.
  Legacy config and state-dict keys are unchanged.
- Ran router-only one-step smoke from existing Route A 10k weights at
  `artifacts/adaptive_lookahead/20261109_router_smoke_v2/run.sh`. Loaded checkpoint,
  verified only `0.74M / 3.72B` parameters trainable, used t=1 intent, completed
  backward and saved step1. Utility loss `0.7047`, target `0.5243`, predicted
  probability `0.4448`, grad norm `0.14`, step time `12.10 s`. This is a code
  smoke; one step does not train a useful router or establish task benefit.
- Focused tests after the code change: 48 passed across router freeze/gradients,
  intent selection, utility objectives, legacy Scheme4, partial expert,
  serving config and evaluator CLI contracts. `compileall` and fatal Ruff rules
  pass. Full lint reports 35 existing findings in the large proposal/config
  files; changed files now pass Ruff formatting.
- Final v2 router-only smoke under
  `artifacts/adaptive_lookahead/20261109_router_smoke_v2/` was verified against
  its Route A initialization: 1,143 shared tensors remained exactly unchanged;
  only 22 new router tensors (737,089 parameters) were added. Its single
  optimizer update at LR `9.998e-9` is a contract smoke, not a fitted router.
- The first serving smoke exposed a stale strict diagnostics schema. The
  consumer now accepts both optional bounded routing fields while retaining
  legacy payload support. After the fix, a Spatial task 0 episode at environment
  seed 7 succeeded in 91 steps and produced 17 diagnostic requests; mean router
  probability was `0.4655` (range `0.3980–0.5425`). Evidence:
  `artifacts/adaptive_lookahead/20261109_eval_smoke_v2/`.
- Started a 1,000-step router-only pilot from the authoritative Route A 10k
  checkpoint, seed42, batch8 x accumulation2, h=10, t=1 intent, 50-step
  warmup and LR5e-5. It is running under
  `artifacts/adaptive_lookahead/20261109_router_pilot/`; no final metrics or
  control claim exists yet.
- Not run: paired no/fixed/soft/hard environment evaluation; multi-seed
  confirmation; multi-horizon; new teacher or world-model distillation.

## 2026-11-10 — pilot archive and request-noise pairing

- Pilot completed 1,000 steps (2:25:50, final loss 0.0752). Freeze verification:
  all 1,143 inherited tensors unchanged; 737,089 router parameters added.
- Evaluated 80 pilot episodes: Spatial all modes 10/10; Long none 5/10,
  fixed/soft/hard 6/10. No adaptive gain. Request RNG was not paired.
- Added default-off deterministic request noise, seed/hash acknowledgments,
  action hashes and strict summaries. 10 further seeded episodes, zero errors;
  fixed repeat exactly reproduces 70 requests; no-lookahead residual zero.
- 62 focused tests pass; compileall/format/fatal lint pass. Five old full-lint
  findings remain in LIBERO main. No dependencies installed or user files removed.
- Exact changes/commands/results:
  `docs/journal/2026-11-10_router_pilot_and_request_noise.md`.

## 2026-11-10 — frozen utility diagnosis and router gate

- Completed the read-only v3 diagnosis on physical RTX 5090 GPU1: 160 frozen
  forwards over 80 frames / 40 tasks, 71.1 seconds and 7.58 GiB peak allocated.
  Strict checkpoint load and dataset episode/frame/task alignment passed.
- The prospective partition is not held out for this checkpoint: the router
  pilot trained over the full dataset. Fixed h10 lowers the one-step action-MSE
  proxy from `0.010547` to `0.009480`, but router probabilities remain
  `0.5033–0.5155`; hard routing selects fixed h10 on every row. Router BCE/Brier
  are worse than the development-mean constant on the prospective partition.
- Decision: do not spend more compute on router-only training, horizon sweep or
  teacher distillation until a separately trained checkpoint with whole
  trajectories excluded produces utility signal beyond a constant gate and
  paired rollout validates the proxy. This diagnosis does not establish a
  control result.
- v3 manifest, launcher, log, 160 raw rows and summary are retained under
  `artifacts/adaptive_lookahead/20261110_utility_diagnostic_v3/`. v1/v2 failed
  attempts remain intact as implementation evidence.
- Tests: 62 focused serving/router/request-noise/utility tests passed; Ruff and
  both relevant Bash syntax checks passed. No weights, source model path,
  dependency environment or user data were changed.

## 2026-11-10 — Independent utility gate

- Added `scripts/fit_scheme4_utility_gate.py` as an isolated experiment runner;
  production model, data loader, trainer, config and serving paths remain
  unchanged. It samples whole trajectories by the existing hash split, caches
  only pooled policy-available inputs in RAM, fits a fresh existing router on
  action-loss quadratic coefficients, saves/reloads it before validation, and
  evaluates actual projected action MSE.
- The CPU suite covers trajectory isolation and unpadded action targets,
  exact router pooling replay, detached inputs/targets, correct quadratic
  utility arithmetic, parameter gradients, deterministic tiny-set overfit,
  episode-paired bootstrap and bad-input rejection. All `22` focused
  regression tests passed after the manifest-directory guard adjustment.
  Ruff, format, compileall and launcher syntax passed.
- The four-forward/two-update GPU smoke strictly loaded the real checkpoint,
  verified data row alignment and standalone router save/reload, and passed
  with unchanged frozen weights. Its evidence is retained separately at
  `artifacts/adaptive_lookahead/20261110_independent_utility_smoke_v1/`.
- Full A5 completed without errors: 640 training and160 validation draws,
  200 fixed updates, strict frozen-parameter boundary, RTX 5090 GPU1, 459.5s,
  peak7.58GiB. It failed the declared gate: a fresh router mostly selected
  full fusion and its actual one-step action MSE was slightly worse than fixed
  h10. No further router fit or rollout follows from this evidence.
- Exact protocol, hashes, rows, weights and output are in
  `artifacts/adaptive_lookahead/20261110_independent_utility_gate_v1/`; details
  are in `docs/journal/2026-11-10_independent_utility_gate.md`.

## 2026-11-10 — C1 fusion-scale evidence recheck

- Re-ran the existing final evidence audit from this workspace:
  `.venv/bin/python scripts/aaai27/pi/audit_active_goal_p0_evidence.py --root . --require-complete`.
  Exit code `0`; all five gates completed, with no pending or invalid evidence.
  C1 passed 193 archive hashes, exact episode-key coverage, six-scale curve
  recomputation and all 27 paired-statistic checks.
- Confirmed the serving contract in `pi0_proposal_variants_pytorch.py`:
  fusion scale multiplies only the future delta suffix residual. Scale zero
  keeps the spatial auxiliary prefix and exactly zeros the future action
  residual. This makes C1 a fixed-checkpoint future-residual ablation.
- Existing C1 result: scale 1.00 versus 0.00 is `+3.331 pp` Spatial,
  `+8.257 pp` LIBERO-10 and `+5.852 pp` over these two suites combined.
  Episode-bootstrap 95% intervals are positive for all three comparisons.
  The raw curve increases on both suites from scale 0.00 to 1.25, so it does
  not reproduce the previously stated fixed-scale Spatial–Long trade-off.
- Boundary: one trained checkpoint/seed and evaluation seed7; bootstrap is
  episode-level, not training-seed uncertainty. Scale 0.00 is not the separate
  trained delta-disabled baseline (`92.21/83.17/86.75`). No new training,
  rollout, or code test was run in this read-only verification step.
- Existing alpha-zero versus trained delta-disabled rows were additionally
  paired across exact task/key/seed matches. Delta-disabled wins by `+1.665 pp`
  Spatial (McNemar `p=0.00237`, paired bootstrap CI `[+0.624,+2.706]`) and
  `+11.195 pp` LIBERO-10 (`p=5.91e-34`, CI `[+9.408,+12.981]`). C1 alpha-one
  outcomes match Route A exactly per key. The checkpoint hashes and trainable
  configurations differ, so this is not a residual-only causal comparison.
- Result details are in `docs/EXPERIMENT_RESULTS.md`; raw archive is
  `artifacts/aaai27/pi/local_archives/hypothesis_c_fusion_scale_c1_local_recovery_v1_20260731/`.

## 2026-11-10 — C1 fixed-scale cross-seed replication queued

- Predeclared a same-checkpoint `alpha=1` versus `alpha=0` comparison across
  training seeds42/43/44. Seed42 has completed C1 data; existing seed43/44
  alpha-one JSONL outputs are reused. Only missing alpha-zero evaluations are
  run, one on each RTX 5090, using the same suites, episode keys and eval seed7.
- No weights are trained and no scale is selected from these results. Report
  each seed separately, plus mean and standard deviation across training seeds;
  paired episode bootstrap remains within-seed uncertainty only.
- Protocol, checkpoint paths, immutable output prefixes and launcher are in
  `artifacts/adaptive_lookahead/20261110_c1_seed_replication/`.

## 2026-11-10 — C1 replication EGL startup diagnosis

- The seed43/44 alpha-zero v2 runs started policy servers but produced zero
  episode rows. LIBERO clients failed while creating the first off-screen
  environment because sandboxed processes could not access the NVIDIA driver;
  EGL enumerated zero devices and rejected the configured device IDs.
- Outside the sandbox both RTX 5090s were visible, and short EGL context
  initialization passed on physical device IDs 0 and 1. The v2 logs are kept.
- A regular-path v3 retry reproduced the same zero-device error despite both
  policy servers loading. The failure is preserved under v3 prefixes; the
  next attempt must execute through the GPU-authorized command path. Protocol
  and checkpoints remain unchanged.

## 2026-11-10 — C1-MS authorized v4 execution and CPU finalizer

- Both seed43/44 v4 policy services and six Spatial shards now run through
  the authorized GPU command path (session59486). At13:34:31Z, observed175/181
  episode rows with zero errors/duplicates; these are partial progress only.
- Corrected stale v2 postprocessing using a separate CPU consumer, session76658.
  Exact live launcher bytes are preserved as `run.launched_v4.sh`; the future
  launcher was atomically corrected without changing its running descriptor.
- Added strict three-seed summary code/tests:17 targeted tests, Ruff
  check/format, Bash syntax and incomplete-result rejection pass. No model,
  checkpoint, environment or scientific protocol change was made.
- Full rationale, commands and remaining boundaries:
  `docs/journal/2026-11-10_c1_seed_replication.md`. Reports will be written to
  `artifacts/adaptive_lookahead/20261110_c1_seed_replication/analysis_v4/`
  only after all four endpoints validate. No new complete result exists yet.

## Current Next Action

Adaptive-router, multi-horizon and teacher experiments remain paused. The
one-step offline utility target did not establish a state-dependent advantage
over fixed h10, so a paired rollout would not answer a passed predeclared
hypothesis. C1 provides a clean same-checkpoint residual-scale result but does
not support the expected Spatial–Long trade-off. Do not spend more GPU time on
router tuning, multi-horizon prediction, or teacher distillation from these
results alone. The current bounded experiment is the predeclared fixed-scale
replication across training seeds42/43/44; the next architecture decision
depends on its per-seed results.
