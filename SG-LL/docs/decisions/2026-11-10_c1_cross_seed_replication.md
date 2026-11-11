# C1 fixed-checkpoint cross-training-seed replication

- `CONFIRMED`: The existing seed42 C1 archive passes the previously recorded
  five-gate evidence audit. Its alpha1-minus-alpha0 differences are +3.331 pp
  Spatial and +8.257 pp LIBERO-10. Unlike the separately trained
  delta-disabled control, alpha0 retains the learned spatial prefix and
  disables only the final future suffix residual. Source:
  `docs/EXPERIMENT_RESULTS.md` and the C1 local-recovery archive it cites.
- `PROPOSED`: Check whether that same-checkpoint intervention has a consistent
  direction across independently trained seeds42/43/44. The protocol was
  declared in `artifacts/adaptive_lookahead/20261110_c1_seed_replication/manifest.json`
  before successful evaluation. Reuse existing alpha1 records; evaluate only
  missing alpha0 arms for seeds43/44. No retraining or scale/checkpoint search.
- `CONFIRMED`: Protocol remains Spatial2402 and LIBERO-10 2519, one initial
  state/trial per task, environment seed7, three workers per seed, h10, normal
  predicted latent, unchanged preprocessing/flow steps, no video. Raw outcomes
  are paired by exact task/episode/environment-seed keys and task strings.
- `CONFIRMED`: The reused alpha1 outputs predate request-noise pairing. Do not
  claim common per-request noise draws. Seed42 uses its audited archive;
  retain execution-site differences as a limitation. The comparison cannot
  separate all hardware/runtime randomness from training-seed variation.
- `PROPOSED`: Report each seed and suite, retained within-seed 10,000-resample
  episode-bootstrap intervals and McNemar statistics, and mean ± sample SD
  across the three training seeds. The combined score covers two suites only,
  not full four-suite LIBERO-Plus Aggregate. No across-seed significance claim
  follows merely from within-seed intervals.
- `CONFIRMED`: The action-MSE adaptive utility gate failed independently;
  router tuning, new horizons and teacher expansion remain paused under
  `2026-11-10_independent_utility_gate_result.md`. Partial C1-MS outcomes must
  not select another model or scale. Review complete results before the next
  architecture decision.

Operational recovery and tests are documented in
`docs/journal/2026-11-10_c1_seed_replication.md`; they change no model or
scientific evaluation settings.
