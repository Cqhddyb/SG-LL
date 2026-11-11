# SG-LL Improvement Results

This file contains only executed results. Planned experiments are listed in
`docs/EXPERIMENT_MATRIX.md` and are not copied here as if completed.

## Existing authoritative baseline

Source: `docs/shared/2026-07-23_delta_disabled_four_suite_result.md`.

| Method | Spatial | Object | Goal | Long/LIBERO-10 | Aggregate |
|---|---:|---:|---:|---:|---:|
| Route A fixed SG-LL | 93.88% | 90.31% | 80.93% | 80.23% | 86.21% |
| Delta-disabled control | 92.21% | 90.39% | 81.63% | 83.17% | 86.75% |

The aggregate Route A difference is -0.538 percentage points and is not a
claim of overall improvement. Route A is significantly better on Spatial and
significantly worse on LIBERO-10 in this paired comparison.

## Existing fixed-checkpoint fusion-scale evidence

The C1 archive was independently re-audited in this workspace with
`.venv/bin/python scripts/aaai27/pi/audit_active_goal_p0_evidence.py --root . --require-complete`.
The read-only audit exited successfully with `5 complete / 0 pending / 0 invalid`;
for C1 it verified 193 file hashes, six scales, 18 curve rows, 27 paired
comparisons, exact episode keys, and checkpoint model SHA256
`c141110dd2e67d32dffe6721044e6f0fe9c1dc18ab12c5b2d0c5d1cf421f4766`.

This is an existing inference-only intervention on one Route-A seed42, 10k
checkpoint. Evaluation used LIBERO-Plus Spatial (2,402 episodes) and
LIBERO-10 (2,519 episodes), one trial per task, environment/evaluation seed 7,
three workers, future offset 10, and video disabled. The only intervention was
the final future-residual fusion scale. At scale 0, serving sets the future
residual to zero while retaining the spatial auxiliary prefix; it is not the
separately trained delta-disabled checkpoint in the baseline table above.

| Future-residual scale | Spatial | LIBERO-10 | Two-suite aggregate |
|---:|---:|---:|---:|
| 0.00 | 90.550% (2175/2402) | 71.973% (1813/2519) | 81.040% (3988/4921) |
| 0.25 | 92.173% (2214/2402) | 74.474% (1876/2519) | 83.113% (4090/4921) |
| 0.50 | 92.714% (2227/2402) | 76.459% (1926/2519) | 84.393% (4153/4921) |
| 0.75 | 92.923% (2232/2402) | 79.039% (1991/2519) | 85.816% (4223/4921) |
| 1.00 | 93.880% (2255/2402) | 80.230% (2021/2519) | 86.893% (4276/4921) |
| 1.25 | 93.422% (2244/2402) | 82.612% (2081/2519) | 87.889% (4325/4921) |

Paired comparison of scale 1.00 against scale 0.00 gives `+3.331` percentage
points on Spatial (95% episode-bootstrap CI `[+2.206,+4.455]`, McNemar
`p=5.998e-9`), `+8.257` points on LIBERO-10 (CI `[+6.550,+9.964]`,
`p=3.832e-21`), and `+5.852` points on the two-suite aggregate (CI
`[+4.816,+6.889]`, `p=9.251e-29`). The difference sign is `scale 1.00 minus
scale 0.00`. These episode-level intervals do not estimate training-seed
variation.

The raw curve does not support the earlier fixed-scale trade-off hypothesis:
from scale 0.00 to 1.25, success rises on both suites. This result is specific
to this checkpoint and evaluation manifest, uses one training seed and one
environment seed, and is not a multi-seed confirmation or a full four-suite
Aggregate result. It also does not establish that an adaptive router improves
control. The immutable archive and raw paired evidence are under
`artifacts/aaai27/pi/local_archives/hypothesis_c_fusion_scale_c1_local_recovery_v1_20260731/`.

### Cross-check against the separately trained delta-disabled policy

The C1 `alpha=0` policy was also paired against the existing independently
trained delta-disabled policy using only retained per-episode CSVs. For both
suites, the `(task_id, episode_id, eval_seed)` key sets and task strings match
exactly; the C1 `alpha=1` success bit matches the Route-A success bit on every
key. Sign below is `trained delta-disabled minus C1 alpha=0`.

| Suite | C1 alpha=0 | Trained delta-disabled | Paired difference | Discordant wins: control / C1 alpha=0 | McNemar exact p | 95% paired episode-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Spatial | 2175/2402 (90.550%) | 2215/2402 (92.215%) | +1.665 pp | 103 / 63 | 0.00237 | [+0.624, +2.706] pp |
| LIBERO-10 | 1813/2519 (71.973%) | 2095/2519 (83.168%) | +11.195 pp | 421 / 139 | 5.91e-34 | [+9.408, +12.981] pp |

The bootstrap used 10,000 episode resamples with seed `20261110`. This
comparison uses the same evaluation episodes and seed7, but it compares
different trained checkpoints: Route-A SHA256
`c141110dd2e67d32dffe6721044e6f0fe9c1dc18ab12c5b2d0c5d1cf421f4766` versus
delta-disabled SHA256
`4334bc8b4c08d8ffc65ac119148436d7d32e01cf9feb6b5650137a01718774e6`. The
delta-disabled model also has a different trainable-parameter configuration.
This reanalysis therefore shows that inference-time `alpha=0` on a
future-trained checkpoint is not the same policy as a model trained with the
future branch disabled; it cannot attribute their score gap to the inference
residual alone. Existing source rows are the C1
`alpha100_vs_alpha000_{libero_spatial,libero_10}_per_episode.csv` and the
delta-disabled archive's `paired_route_a10k_vs_delta_disabled_{libero_spatial,libero10}_per_episode.csv`.

The next planned check is a cross-training-seed replication of the fixed
inference intervention: use existing seed43/44 alpha=1 episode records and add
only alpha=0 runs for those same two checkpoints and suites. Seed42 C1 already
provides the third training seed. The fixed comparison will be summarized per
training seed; episode resampling will not replace training-seed variability.
Because the reused alpha-one runs predate the opt-in request-noise protocol,
this replication is paired by exact task/episode/environment-seed keys, not by
per-request action-noise hashes. It will not claim common-random-number pairing.
The initial launch stopped before evaluation because the server subprocess
did not inherit this workspace's `src` import path. The corrected v2 servers
started, but LIBERO clients then failed at first environment construction:
the sandbox could not access the NVIDIA driver, EGL enumerated zero devices,
and no episodes ran. Outside the sandbox both RTX 5090s and device-specific
EGL context smoke tests passed. Failed logs remain preserved; the authorized
GPU retry uses fresh `*_20261110_v4` prefixes. A regular-path v3 retry also
failed before the first episode with the same zero-EGL-device error and is
preserved separately.

The explicitly GPU-authorized v4 retry is now running on both RTX5090s. At
`2026-11-10T13:34:31Z`, the retained snapshot contains175/2402 Spatial episodes
for seed43 and181/2402 for seed44, with zero errors/duplicates. LIBERO-10 is
pending behind Spatial. These are partial progress counts, not performance
estimates. A CPU-only finalizer waits for all four complete endpoints, checks
exact keys/task strings/source hashes and writes the per-seed and three-seed
summary to `artifacts/adaptive_lookahead/20261110_c1_seed_replication/analysis_v4/`.
The original launched script's stale v2 postprocessing is preserved and
superseded by this consumer; no GPU evaluation was restarted for that fix.
Journal: `docs/journal/2026-11-10_c1_seed_replication.md`.

## New experiments

### 2026-11-06: CUDA works; one-step training failed before forward

- **Executed diagnostic:** `.venv` PyTorch `2.11.0+cu128` detects RTX 5090
  outside the sandbox and produces a finite CUDA matrix multiplication.
- **Executed attempt:** fixed SG-LL h=10, official π0.5 initialization requested,
  B1, seed42, zero workers, one requested optimizer step, physical GPU 1.
  Exact command and failure log:
  `artifacts/adaptive_lookahead/20261106_gpu_smoke/{run.sh,train.log}`.
- **Data result:** existing LIBERO-Plus normalization loaded and the real
  2,238,036-row dataset split built in a project-local writable cache.
- **Failure:** OOM while transferring the newly constructed model to CUDA,
  before checkpoint weights were loaded or a training forward began. A separate
  `airbot-pi0.5` process, PID 74505, occupied about 29 GB on each GPU by then.
  Initial free-memory readings therefore did not establish an exclusive GPU.
- **Completed training steps: 0. Saved checkpoints: 0. Evaluation episodes: 0.**
  No loss, backward, throughput, peak-training-memory or success-rate result
  exists for this attempt. The empty output directory was removed; the failure
  evidence and reusable 214 MB dataset cache remain.

This attempt predates the successful November 9 execution and is retained only
as infrastructure-failure evidence.

### 2026-11-09: GPU smokes, adaptive-serving smoke, and router pilot

**Fixed h=10 smoke**

- Device: physical RTX 5090 0; `.venv` PyTorch `2.11.0+cu128`; seed42;
  LIBERO-Plus real loader; B1; one optimizer step; offset/horizon 10.
- Loaded official π0.5 Torch weights, VGGT and existing h=10 tokenizer.
- Result: loss `0.0862`, flow loss `0.0158`, current geometry loss `1.0566`,
  future reconstruction loss `0.6961`, grad norm `0.13`, peak allocated memory
  `12.78 GB`, optimizer step time `18.32 s`. Saved step1 checkpoint at
  `checkpoints/pi05_libero_plus_geo/sgll_fixed_h10_gpu_smoke_20261109/1`.
- Command and full output: `artifacts/adaptive_lookahead/20261109_gpu_smoke/`.

**Router-only contract smoke**

- Device: RTX 5090 0; seed42; B1; one step; initialized from existing Route A
  checkpoint `scheme4_lac_double_align_plus_official_frozen_b16_cg006_fut001_tok0_h10_10k/10000`.
- Only `0.74M / 3.72B` model parameters were trainable (utility router).
  Training used an extra detached action-expert suffix pass with the same
  initial Gaussian noise and `t=1` as serving. The flow-fusion gate was detached,
  so only the action-loss utility target trains router weights.
- Result: total loss `0.0830`, flow loss `0.0125`, base flow loss `0.0156`,
  utility loss `0.7047`, utility benefit proxy `0.0049`, target `0.5243`,
  router probability `0.4448`, grad norm `0.14`, step time `12.10 s`. Saved
  step1 checkpoint at
  `checkpoints/pi05_libero_plus_geo/sgll_router_smoke_route_a_seed42_20261109/1`.
- Exact command/log: `artifacts/adaptive_lookahead/20261109_router_smoke_v2/`.
- A source tensor comparison against its initialization found all 1,143 common
  tensors unchanged, no missing tensors, and only 22 new router tensors
  (737,089 parameters); see `freeze_verification.json`.

**One-episode served-policy smoke**

- The first attempt reached the environment but failed at request logging
  because the diagnostics consumer rejected the new Router fields. The exact
  failed row is preserved under `artifacts/adaptive_lookahead/20261109_eval_smoke/`.
- The consumer was extended with an optional strict adaptive schema and tests.
  Rerunning the same Spatial task 0 at environment seed 7 succeeded in 91
  environment steps (17 policy requests). Router probability ranged
  `0.3980–0.5425`, mean `0.4655`. Exact outcome and request logs are under
  `artifacts/adaptive_lookahead/20261109_eval_smoke_v2/`.
- The evaluated checkpoint had one optimizer step at LR `9.998e-9`; this is
  only serving/evaluator evidence and does not show Router utility or improved
  control.

**Router-only pilot (completed; no demonstrated adaptive gain)**

- Predeclared 1,000 optimizer steps from the authoritative Route A checkpoint
  `scheme4_route_a_condneg_plus_official_frozen_b8accum2_cg006_fut001_tok0_h10_10k/10000`,
  training seed42, dataset `Sylvest/libero_plus_lerobot`, micro-batch8 x
  accumulation2, effective batch16, h=10, `t1_noise`, utility coefficient
  0.1, warmup50 and constant peak LR5e-5.
- Completed 1,000 steps in 2:25:50; final total loss `0.0752`. Step975 utility
  BCE `0.6931`, mean benefit `0.0009`, target `0.5044`, probability `0.5041`.
  These are training means, not held-out utility evidence.
- Final checkpoint:
  `checkpoints/pi05_libero_plus_geo/sgll_router_pilot_route_a_seed42_1000_20261109/1000`.
  All 1,143 inherited tensors remain bitwise unchanged; exactly 22 router
  tensors / 737,089 parameters were added. Command, log and freeze report:
  `artifacts/adaptive_lookahead/20261109_router_pilot/`.

The fixed and Router contract runs are one-step implementation smokes. The
one-episode result validates the chain but is not a rate estimate. The
1,000-step pilot is complete; no adaptive control benefit is established.
The Router contract checkpoint was run with the final flow-gate
stop-gradient behavior.

### 2026-11-10: pilot evaluation and request-noise validation

Same pilot checkpoint, training seed42; environment seed7; task indices0–9,
initial-state index0, one trial each, two client shards per suite/mode:

| Mode | Spatial | Long/LIBERO-10 |
|---|---:|---:|
| none | 10/10 | 5/10 |
| fixed h10 | 10/10 | 6/10 |
| soft | 10/10 | 6/10 |
| hard | 10/10 | 6/10 |

`paired_pilot_v1` validates all 80 exact keys, no duplicates/errors. All 941
none requests have exactly zero hidden residual and velocity correction in
all ten flow steps. Soft probabilities average ~0.507/0.508; hard never
closes Spatial and closes only6/701 Long requests. Soft versus fixed Long
paired difference is0, bootstrap95% CI[-50,+50] percentage points, McNemar1.
This is a low-power pilot, one training seed, not full LIBERO-Plus Aggregate.
Server-start seed7 did not pair individual action-noise draws across clients.

Added a default-off request-key noise protocol, documented in
`docs/decisions/2026-11-10_request_noise_pairing.md`. Bounded GPU integration
run `request_noise_smoke_v1`: two suites × task0 × four modes, eight episodes,
zero errors. Spatial succeeds in all modes; Long none fails and fixed/soft/hard
succeed. This is integration evidence only, not a policy selection result.
Common request-noise hashes match across modes (122 unique request keys).
All 120 none requests have exact zero residual/correction. Two fixed repeats
also succeed: 18 Spatial and52 Long requests reproduce exact noise hashes,
returned-action hashes, environment steps and diagnostics. No model weights,
flow-step count, preprocessing or existing evaluation default changed.

Reproduction:
`bash artifacts/adaptive_lookahead/20261110_router_paired_eval/request_noise_smoke.sh`
(immutable run directories; do not overwrite). Summaries and raw evidence:
`artifacts/adaptive_lookahead/20261110_router_paired_eval/{results,diagnostics,summaries}/`.
The summary script checks keys, errors, request counts, no-lookahead zeros,
request seeds/hashes and fixed repeat equality; source JSONL SHA256s are saved.

No new teacher, world-model distillation, multi-horizon or multi-training-seed
result exists. The per-sample utility diagnostic below is complete, but its
prospective partition is not held out for the current pilot because that
checkpoint trained on the full existing train split.

### 2026-11-10: frozen utility diagnosis (training proxy only)

- Checkpoint: router-only seed42/1,000-step pilot at
  `checkpoints/pi05_libero_plus_geo/sgll_router_pilot_route_a_seed42_1000_20261109/1000`.
  The model was strictly loaded and fully frozen; no weights or optimizer state
  were written.
- Protocol: 80 selected midpoint frames across 40 tasks, one trajectory per
  task in each development/prospective partition, two draws per frame (160
  forwards). The SHA256 split reserved 1,446 whole trajectories prospectively.
  The checkpoint trained on the full data split, so this is not held-out
  validation. Each draw varies augmentation, noise and flow time together.
- Runtime: physical RTX 5090 GPU1, 71.1 seconds, 7.58 GiB peak allocated.
  All sampled indices matched the underlying LeRobot dataset. Raw evidence:
  `artifacts/adaptive_lookahead/20261110_utility_diagnostic_v3/`; manifest
  SHA256 `d0dc3a5f3a0f03a0c78fa72853581c5c2de499f9b93b00068f1362cbe0076fcf`,
  rows SHA256 `c16eabed510fb6d5c90a86a6af3cc65ab02ac8e02d0cd7af103e6175175e40de`.
- On the prospective partition, none/fixed/half/soft action MSE was
  `0.010547 / 0.009480 / 0.009877 / 0.009868`; 83.75% of row-level fixed-minus-
  none benefits were positive, with mean benefit `0.001067`. These are
  one-step action-loss proxies, not rollout success rates.
- Router probabilities were nearly constant (`0.5033–0.5155`; mean `0.5086`).
  Router BCE/Brier were `0.693130/0.0001828`, worse than a development-mean
  constant at `0.693091/0.0001634`; hard routing selected fixed h10 for every
  row. No state-dependent utility or closed-loop gain is established.
- Decision: stop router-only expansion and defer horizon/teacher experiments.
  Reopen only with an independently trained checkpoint, whole-trajectory
  exclusion and a utility signal that beats a constant baseline, followed by
  paired closed-loop validation.
- Verification: 62 focused serving/router/request-noise/utility tests passed;
  Ruff and launcher syntax checks passed. No extra rollout or training was
  run after this gate failed.

### 2026-11-10: independent utility-router gate (one-step proxy)

- Checkpoint: fixed SG-LL/router pilot seed42 at
  `checkpoints/pi05_libero_plus_geo/sgll_router_pilot_route_a_seed42_1000_20261109/1000`.
  Its pretrained backbone and planning modules were frozen; the existing
  737,089-parameter utility router was freshly initialized and was the only
  trainable module. This is an independent utility fit, not an independent VLA
  policy or training-seed replication.
- Protocol: 40 LIBERO-Plus tasks, 8 development trajectories/task for fitting
  (320 trajectories, 640 draws), 2 reserved trajectories/task for validation
  (80 trajectories, 160 draws), two independently seeded noise/time/augmentation
  draws per midpoint. All action chunks are unpadded. Whole trajectory IDs
  from the prior prospective partition were excluded from router fitting; the
  prior validation trajectory/task was also excluded from fresh validation.
  The base checkpoint had seen every trajectory. Full IDs, indices, seeds and
  hashes are bound by the immutable manifest.
- Training: 200 AdamW updates, batch64, seed42, LR5e-5, 50-step linear warmup,
  weight decay0.01. Only training rows set the utility-normalization constant
  and comparison gates. The training-optimal constant gate was1.0; the BCE
  target-mean constant was0.50498. Router weights were saved and strictly
  reloaded before validation access.
- Validation one-step normalized action MSE: none `0.0129883`, fixed h10
  `0.0119928`, soft router `0.0119957`, hard router `0.0119928`, half `0.0123707`,
  training-optimal constant `0.0119928`, and BCE target-mean constant
  `0.0123657`. Router probability was `0.9888–0.9967` (mean `0.9944`), with
  correlation `-0.030` to per-row fixed-minus-none benefit. Fixed improved
  over none on80.6% of draws, but soft routing did not improve on fixed.
- Paired bootstrap over80 validation episodes, averaging two draws within each
  episode: fixed-minus-soft `-2.97e-6`, 95% CI `[-4.94e-6,-1.09e-6]`; half-minus-
  soft `+3.75e-4`, CI `[+2.03e-4,+5.54e-4]`. The sign convention is
  `control loss - soft-router loss`. The predeclared adaptive gate failed.
- Runtime/compute: RTX 5090 GPU1, 459.5 seconds, peak allocated memory
  7.58 GiB, 737,089 router parameters. No rollout, multi-horizon run, new
  teacher, data/cache, production model, or dependency was added.
- Decision: stop this adaptive utility path. The evidence says the tested
  one-step action-loss utility is dominated by a constant fixed-h10 decision
  on this frozen policy; it does not disprove every possible control utility or
  establish closed-loop performance. Do not tune or launch a paired rollout
  from this failed gate. Evidence lives in
  `artifacts/adaptive_lookahead/20261110_independent_utility_gate_v1/` and its
  predeclared protocol is in
  `docs/decisions/2026-11-10_independent_utility_gate.md`; terminal decision:
  `docs/decisions/2026-11-10_independent_utility_gate_result.md`.

Reproduction: `bash artifacts/adaptive_lookahead/20261110_independent_utility_gate_v1/run.sh`.
Focused tests:
`PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest -q scripts/fit_scheme4_utility_gate_test.py scripts/diagnose_scheme4_utility_test.py src/openpi/models_pytorch/scheme4_adaptive_lookahead_test.py`.

