# SG-LL

Core implementation of **Spatially Grounded Latent Lookahead for
Vision-Language-Action Models**.

- [`SG-LL/`](SG-LL/): pi0.5 model extensions, training, serving, evaluation,
  tests and research documentation. Start with its [README](SG-LL/README.md).
- `LIBERO-plus/` and `openvla-oft/`: pre-existing repository content, preserved
  by this source synchronization.

Adaptive lookahead is experimental: its current single-horizon utility router
has not demonstrated a control improvement. Multi-seed fixed-residual
evaluation is still in progress. See the
[recorded results and limitations](SG-LL/docs/EXPERIMENT_RESULTS.md).
