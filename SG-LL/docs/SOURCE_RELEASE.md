# Core-source release boundary

This snapshot adds `SG-LL/` to the existing `Cqhddyb/SG-LL` repository. Existing
`LIBERO-plus/` and `openvla-oft/` files and the remote parent commit are preserved.

Included: model and training code, data/config pipeline, client and evaluator,
source tests, offline cache/tokenizer tools, selected statistical utilities,
dependency manifests/lock, and focused method/experiment documentation.

Excluded: checkpoints, datasets, normalization/feature caches, raw results and
logs, environments, agent memory, and machine-specific queue orchestration.
Documented results refer to local evidence; they are not newly reproduced by
publishing this source tree. Ongoing evaluations are not final results.

The only release-copy adaptations to existing Python files are:

1. Six registered VGGT path defaults use `checkpoints/vggt`; the corresponding
   CALVIN config-test assertion matches that default. Explicit path overrides
   remain available.
2. The LIBERO parallel evaluator defaults to `../LIBERO-plus` instead of a
   personal absolute path. Supply `--libero-pythonpath` for other installations.
3. A proposal-model docstring no longer contains a personal directory name.

Local absolute paths in selected Markdown records become `<SGLL_ROOT>`,
`<WORKSPACE>` or `<REMOTE_WORKSPACE>` placeholders. The source `.gitignore`
is extended to exclude credential files, agent directories and tool caches.
The dependency lock, algorithm, parameter names, model tensors, preprocessing,
losses, horizon values, evaluation counts and random-seed policy are preserved.
The active GPU experiment's originals were not edited.

The release README provides the verification command. CPU tests use the
existing pi0.5-compatible environment; a clean dependency install, GPU model
reload and new environment rollouts are not part of source publication.
