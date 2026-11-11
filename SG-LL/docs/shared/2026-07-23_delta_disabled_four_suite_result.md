# 2026-07-23 Delta-Disabled Four-Suite Result

## Result Status

`CONFIRMED`: The trained delta-disabled control is now complete on all four
clean LIBERO-Plus suites. Every Route A/control comparison has exact matched
episode keys, zero duplicate keys, and zero error rows. The read-only P0
registry accepts this claim as complete; it reports `3 complete`, `6 pending`,
and `0 invalid` claims overall.

## Matched Results

| Suite | Route A | Trained delta-disabled | Route A minus control | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Spatial | 2255/2402 (93.88%) | 2215/2402 (92.21%) | +1.665 pp | 0.001704 | [+0.666, +2.706] pp |
| Object | 2274/2518 (90.31%) | 2276/2518 (90.39%) | -0.079 pp | 0.9382 | [-1.072, +0.913] pp |
| Goal | 2097/2591 (80.93%) | 2115/2591 (81.63%) | -0.695 pp | 0.2684 | [-1.853, +0.463] pp |
| LIBERO-10 | 2021/2519 (80.23%) | 2095/2519 (83.17%) | -2.938 pp | 0.000445 | [-4.565, -1.270] pp |
| Aggregate | 8647/10030 (86.21%) | 8701/10030 (86.75%) | -0.538 pp | 0.09237 | [-1.157, +0.080] pp |

## Interpretation Boundary

- Route A remains significantly better than this trained control on the
  spatial suite only.
- The aggregate difference is not statistically significant and is
  directionally unfavorable to Route A; LIBERO-10 significantly favors the
  trained delta-disabled control.
- Therefore this control cannot support a claim that Route A consistently or
  aggregate-wise outperforms the trained no-delta architecture.
- This result does not alter the separately validated Route A versus official
  four-suite comparison: Route A remains `8647/10030` versus official
  `8470/10030`, `+1.765 pp`, paired p=`1.88e-07`. It does require paper text
  to keep the two comparisons distinct.
- The trained delta-disabled control has a different trainable-parameter
  configuration from Route A. It is a matched data/step/control comparison,
  not a parameter-identical causal isolation of the transition branch.

## Evidence

- `results/aaai27/pi/summaries/paired_route_a10k_vs_delta_disabled_libero_spatial.json`
- `results/aaai27/pi/summaries/paired_route_a10k_vs_delta_disabled_libero_object.json`
- `results/aaai27/pi/summaries/paired_route_a10k_vs_delta_disabled_libero_goal.json`
- `results/aaai27/pi/summaries/paired_route_a10k_vs_delta_disabled_libero10.json`
- `results/aaai27/pi/summaries/paired_route_a10k_vs_delta_disabled_four_suite.json`
- `artifacts/aaai27/pi/p0_evidence_audit_latest.json`
