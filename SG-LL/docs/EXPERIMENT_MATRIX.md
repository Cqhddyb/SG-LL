# SG-LL Improvement Experiment Matrix

All rows use the same LIBERO-Plus repository, preprocessing, evaluation
manifest, evaluation seed and checkpoint-selection rule unless stated. A row
is marked **planned** until its log and summary exist; no planned row is an
experimental result.

| ID | Question/control | Trainable change | First run | Decision rule |
|---|---|---|---|---|
| B0 | Official π0.5 | none | reuse recorded checkpoint | reference only |
| B1 | matched π0.5 adaptation | reuse recorded adaptation | reuse | reference only |
| B2 | Spatial-only | existing spatial adapter | reuse recorded result | geometry reference |
| B3 | fixed SG-LL h=10 | existing Route A | reuse four-suite record | fixed lookahead reference |
| B4 | trained delta-disabled | future branch disabled | reuse four-suite record | causal no-future control |
| A0 | no-lookahead | same pilot checkpoint, mode `none` | 20 pilot episodes plus2 seeded smoke episodes; exact zero future correction | spatial prefix retained |
| A1 | fixed h=10 | same pilot checkpoint, mode `fixed` | 20 pilot episodes plus2 seeded smoke and2 exact repeats | direct matched reference |
| A2 | soft when-to-look | router only, h=10 | 1000 steps complete; Spatial10/10, Long6/10; utility diagnosis complete | no adaptive gain; proxy router no better than development constant; stop router-only expansion |
| A3 | hard when-to-look | same router, threshold0.5 | Spatial10/10, Long6/10 | nearly always on; no adaptive gain |
| A4 | router ablation | random/frozen router control | not run | skip unless a new utility signal beats the constant baseline |
| A5 | independently fit utility router | fresh router only; frozen VLA/SG-LL | 200 updates; 320 whole train trajectories, 80 held-out-for-router trajectories, 2 draws each | failed: learned gate ~0.994; soft MSE 0.0119957 vs fixed 0.0119928; paired CI against constant below zero; do not launch rollout |
| C1 | fixed future-residual scale control | same Route-A seed42 10k checkpoint; inference-only scale `{0,.25,.5,.75,1,1.25}` | existing LIBERO-Plus Spatial 2402 + LIBERO-10 2519, evaluation seed7 | verified archive; raw success increased on both suites from scale0 to1.25; no fixed-scale Spatial–Long trade-off in this sweep; one training seed, episode-level CI only |
| C1-MS | fixed-scale training-seed replication | existing Route-A seed43/44 10k checkpoints; run only inference scale0; reuse each checkpoint's existing scale1 outputs | v4 running on both RTX5090s; Spatial then LIBERO-10; CPU finalizer queued; no complete result yet | report seed42/43/44 paired scale1-minus-scale0 differences and mean ± sample SD; exact keys/task strings; no scale selection |
| H0 | h=0 explicit branch | no future predictor | CPU contract + eval | exact zero residual |
| H1 | h=5/10/20/40 | new aligned target/cache/tokenizer assets | only if A2/A3 positive | horizon selection is valid |
| W0 | world-only distillation | offline teacher cache | later | future error/control independent gain |
| W1 | geometry+world | joint loss | later | synergy over W0/B2 |

## Statistics and reporting

The adaptive pilot starts with one seed only to catch implementation failures.
Any paper claim uses at least three training seeds, reports per-seed success,
mean ± standard deviation, paired episode differences and bootstrap confidence
intervals. Training seed and environment seed are recorded separately. Latency,
peak memory, parameter count and throughput are reported alongside success.

## Execution order

Current gate (2026-11-10): 1000-step router-only pilot and80 pilot episodes
are complete; no adaptive gain established. A further10 GPU integration
episodes validate optional request-noise pairing and exact fixed replay.
Training seed42, environment seed7, policy seed7; two suites are not full
Aggregate. See `docs/EXPERIMENT_RESULTS.md` and
`artifacts/adaptive_lookahead/20261110_router_paired_eval/summaries/`.

1. CPU contracts and legacy checkpoint/config loading.
2. GPU one-step forward/backward/save smoke.
3. Completed: fixed1000-step A2 pilot and inherited-tensor freeze verification.
4. Completed: 80-episode four-mode pilot and10 request-seeded integration
   episodes. The pilot showed no adaptive gain; utility was not held out.
5. Completed: frozen utility diagnosis on 80 frames / 40 tasks with two draws
   each. Fixed h10 improves the action-loss proxy on average, but the router is
   nearly constant and underperforms a development-mean constant on the
   prospective partition. The checkpoint saw the full training split, so this
   is not held-out evidence. Do not continue router-only training or tune
   against simulator test outcomes.
6. Completed A5: independently fit only the existing router on 640
   action-loss-proxy samples, with 160 samples from 80 reserved trajectories
   for validation. The fresh router converged to almost-always-on and failed to
   beat the training-optimal constant/fixed gate. This holdout excludes only
   router utility training; the pretrained base saw all data. Do not launch the
   proposed closed-loop comparison from this failed gate.
7. Keep A4, multi-horizon and world-distillation experiments paused. Resume
   adaptive work only after a materially new, predeclared control-relevant
   utility target changes the evidence; do not tune against test outcomes.
8. Three-seed confirmation applies only after a non-regressive arm passes that
   gate. Multi-horizon and teacher experiments remain deferred.

The existing C1 fixed-scale sweep is evidence about the effect of scaling the
future residual on one trained checkpoint. It is not the separately trained
delta-disabled control, and it does not validate adaptive routing. See the
paired results and protocol boundary in `docs/EXPERIMENT_RESULTS.md`. The
cross-seed C1-MS run reuses existing scale1 records and evaluates only the
missing same-checkpoint scale0 arm; its immutable protocol and launcher are in
`artifacts/adaptive_lookahead/20261110_c1_seed_replication/`.

