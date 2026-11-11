# Result: independent trajectory-excluded utility gate

- `CONFIRMED`: The preregistered A5 protocol in
  `2026-11-10_independent_utility_gate.md` ran to completion: 640 training
  draws from 320 development trajectories and 160 validation draws from 80
  reserved trajectories, two draws each across 40 tasks.
- `CONFIRMED`: Only a fresh 737,089-parameter router was trainable. The base
  VLA/SG-LL, preprocessing, data, checkpoint and inference path were unchanged.
  Router weights were saved and strictly reloaded before validation access;
  inherited parameter versions remained unchanged.
- `CONFIRMED`: The training-optimal constant gate was `1.0`; validation router
  probability was `0.9888–0.9967` (mean `0.9944`). The learned router mostly
  reproduced fixed fusion.
- `CONFIRMED`: Validation one-step action MSE was none `0.0129883`, fixed
  h10 `0.0119928`, soft router `0.0119957`, half `0.0123707`, and training-mean
  BCE constant `0.0123657`. Fixed-minus-soft paired episode difference was
  `-2.97e-6`, bootstrap 95% CI `[-4.94e-6,-1.09e-6]`; fixed-minus-half was
  `-3.78e-4`. Half-minus-soft was `+3.75e-4`, CI
  `[+2.03e-4,+5.54e-4]`. Thus the declared gate failed.
- `CONFIRMED_BOUNDARY`: Fixed h10 improved this one-step proxy over none by
  `0.000996` on average; 80.6% of validation draws favored fixed. The router
  probability correlation with row benefit was `-0.030`. The base checkpoint
  saw every trajectory, so only router utility fitting was trajectory-excluded.
  No held-out policy-control or multi-seed claim follows.
- `DECIDED`: Stop this adaptive utility path. Do not continue router-only fits,
  tune the gate, add horizons/teachers, or launch a paired rollout from this
  signal. Resume only with a materially new, predeclared control-relevant
  utility target.
- `CONFIRMED`: RTX 5090 GPU1 runtime was 459.5 seconds and peak allocated
  memory 7.58 GiB. Raw manifest, rows, router and log:
  `artifacts/adaptive_lookahead/20261110_independent_utility_gate_v1/`.
