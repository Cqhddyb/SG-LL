# Independent, trajectory-excluded utility gate

- `CONFIRMED`: The previous pilot saw all trajectories. Its soft BCE target
  was concentrated near 0.5 and did not beat a development constant. Evidence:
  `2026-11-10_utility_diagnostic.md` and the v3 artifact summary.
- `DECIDED`: Run one bounded frozen-feature experiment, not another full
  1,000-step VLA pass. Reuse the v3 split seed and all 1,446 excluded episode
  IDs. Select eight development and two reserved trajectories per task by a
  separate versioned hash; exclude the previous diagnostic's validation
  trajectory from the new validation sample. Use one unpadded midpoint and
  two seeded augmentation/noise/time draws per trajectory: 640 training and
  160 validation forwards. Sampling never inspects losses or successes.
- `DECIDED`: Reinitialize the existing router with seed 42. Capture only its
  current-policy inputs (masked spatial mean, language, state, t1-noise intent)
  in RAM. Check pooling equivalence against the real router. No teacher,
  future frame or demonstration action enters the router. Actions are used
  only to compute supervised loss coefficients, as in the previous pilot.
- `DECIDED`: Predeclare a new continuous utility objective. For frozen base
  velocity v0, full-fusion velocity v1, and flow target u, let d=v1-v0,
  q=mean(d²), c=mean((u-v0)d). Train the existing sigmoid gate g on
  mean(q*g²-2*c*g), divided by training mean base MSE only for conditioning.
  This predicts scale utility directly rather than temperature-scaled BCE.
  The quadratic interpolates endpoint velocities and is an approximation
  under TF32; evaluation MUST use the actual suffix fusion and projection.
- `DECIDED`: Fixed seed42, 200 AdamW updates, batch64, LR5e-5 with 50-step
  linear warmup, weight decay0.01. No validation-based early stopping or
  hyperparameter search. Only the router's existing 737,089 parameters train.
  Save standalone router weights and training trace before validation.
- `DECIDED`: Validation compares actual one-step action MSE for none, fixed,
  half, router soft/hard, training-mean BCE target, and the training-optimal
  constant clip(mean(c)/mean(q),0,1). Pair draws within episodes, then bootstrap
  episodes (seed20261110, 10,000 draws). Advance only if the soft router's
  improvement over the training-optimal constant has a strictly positive 95%
  interval and its mean beats fixed. Otherwise stop this router objective;
  do not tune it against these validation rows or simulator test outcomes.
- `CONFIRMED_BOUNDARY`: Only router-utility training excludes the reserved
  trajectories. The pretrained VLA and its fixed SG-LL adapters saw them.
  This cannot establish end-to-end held-out control performance or variability
  across training seeds. Prior diagnostic outcomes informed this design; fresh
  reserved trajectories reduce direct sample reuse but are not a final test.
- `DECIDED`: Preserve production model, loader, trainer, config, checkpoints,
  preprocessing, action dimension/horizon and inference path. Implement the
  bounded experiment in `scripts/fit_scheme4_utility_gate.py` with focused
  tests, a pre-run manifest and launcher. No feature cache or new dependency.
  Do CPU isolation/numerical/gradient/overfit tests and a four-forward GPU
  smoke before the bounded experiment. Do not run new closed-loop episodes
  unless the utility gate passes.

This satisfies the independent-training condition of the previous decision;
it does not supersede its negative result or lift the multi-horizon/teacher gate.
