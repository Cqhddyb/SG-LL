# Decision: explicit request noise for adaptive pilot evaluation

- `CONFIRMED`: The completed `paired_pilot_v1` uses equal environment episode
  keys, but the server seeds Torch only at startup. Two client shards and
  different episode lengths advance the global sampling stream differently.
  Evidence: `scripts/serve_policy_proposal.py::seed_policy_rng`,
  `src/openpi/policies/policy.py::Policy.infer`, and the pilot launcher.
- `CONFIRMED`: Preserve those results under their original protocol; do not
  relabel them as request-noise-matched or compare their absolute rates to the
  new protocol as a model improvement. They are a single-training-seed pilot,
  not a full LIBERO-Plus Aggregate result.
- `CONFIRMED`: Add default-off `--request-noise-policy-seed`.
  Hash the versioned suite/task/initial-state/environment-seed/policy-seed/
  action-chunk-request-index tuple to a 63-bit seed. Exclude mode, run ID, shard
  and worker scheduling. Generate float32 standard normal `[H,A]` noise using
  a private Torch generator on the model device. Pass it through the existing
  explicit-noise inference path; remove request metadata before transforms.
- `CONFIRMED`: Log the acknowledged seed and noise SHA256, and the
  returned action-chunk SHA256. Check common request hashes across modes and
  exact returned-action/diagnostic equality in a repeated fixed-mode smoke.
  This does not guarantee simulator determinism or equal states after policies
  diverge. Only current request metadata reaches inference; no future targets
  or demonstration actions enter the policy.
- `CONFIRMED_SCOPE`: Change no model weights, training objective, dataset,
  preprocessing, flow step count, action horizon or normalization. With no
  request seed the legacy sampling path and response schema remain unchanged.
- `CONFIRMED`: Run only two suites x one episode x four modes, followed by
  a fixed-mode repeat (10 episodes total). Do not enlarge the experiment or
  tune the router on these test outcomes. Diagnose independent validation
  utility before multi-seed training, multi-horizon or teacher changes.

This is an explicit new evaluation seed protocol, not a retrospective change
to any archived baseline. Runtime evidence is recorded in
`docs/journal/2026-11-10_router_pilot_and_request_noise.md`.

Terminal evidence: all 10 bounded episodes completed without errors. The two
fixed-mode repeats reproduce 70 request noise/action hashes and diagnostics
exactly. All 120 no-lookahead requests have zero future residual/correction.
Summaries: `artifacts/adaptive_lookahead/20261110_router_paired_eval/summaries/`.
This validates the protocol, not a control improvement or general determinism.
