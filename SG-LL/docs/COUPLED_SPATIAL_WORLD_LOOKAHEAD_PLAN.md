# Coupled Spatial–World Lookahead Plan

## Scientific sequence

The implementation will proceed from the smallest causal question to the
larger architecture:

1. **Baseline closure.** Reuse the official π0.5, matched adaptation,
   spatial-only, fixed SG-LL and delta-disabled records. Do not retrain a
   baseline whose checkpoint and manifest are already recorded.
2. **Single-horizon adaptive “when to look”.** Keep the h=10 tokenizer,
   spatial adapter, future predictor and π0.5 weights fixed. Train only the
   utility router on the same action-loss proxy. Compare no-lookahead, fixed,
   soft and hard routing on matched episodes.
3. **Action-intent contract.** Align the training and serving intent inputs
   before drawing conclusions from the router. The first deployable condition
   remains the detached action-expert suffix intent; proposal-action conditioning
   is deferred until this test is stable.
4. **Multi-horizon “how far to look”.** Only if single-horizon routing shows a
   positive Long signal, build separate targets/caches/tokenizer support for
   h=5,10,20,40 plus an explicit h=0 branch. The horizon is an environment/data
   frame offset, not an arbitrary latent index.
5. **World representation distillation.** Only after geometry and adaptive
   controls have independent results, add an offline world-teacher cache. Keep
   teacher execution out of deployment and compare geometry-only,
   world-only and joint supervision.

## Minimal implementation choices

- Preserve `Scheme4LACDoubleAlignPI0Pytorch` and legacy fixed h=10 loading.
- Add configuration gates rather than a second model implementation.
- Keep `none` as a true no-lookahead intervention and retain the spatial path.
- Use detached utility labels; report them as an action-loss proxy, never as a
  task-success target.
- Log router probability, selected fraction, fusion scale, fallback ratio,
  future error and latency per evaluation request where the serving path allows.

## Explicit hypotheses

- H1: better geometry improves future-state reconstruction.
- H2: future-state reconstruction improves control only when the predicted
  change is useful for the current action phase.
- H3: selective fusion can retain Spatial gains while avoiding the Long
  degradation of fixed fusion.
- H4: a joint geometry/world representation has an independent benefit after
  parameter and data controls.

H1–H4 are hypotheses, not current results.

## Current dependency gate

The 2026-11-10 independent router utility gate is complete. A fresh router
trained on 640 one-step action-loss proxy draws from 320 development
trajectories and evaluated on 160 draws from 80 router-held-out trajectories
mostly selected full h=10 fusion. Its actual validation MSE was slightly worse
than fixed h=10 and failed the predeclared paired-episode criterion. The base
SG-LL checkpoint had seen the validation episodes, so this does not establish
held-out policy control. Pause router fitting, paired rollout, multi-horizon,
and teacher/distillation work until a materially new, predeclared
control-relevant utility target is available. Keep the existing fixed SG-LL
evidence separate. Details are in `docs/EXPERIMENT_RESULTS.md` and
`docs/decisions/2026-11-10_independent_utility_gate_result.md`.

