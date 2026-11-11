# GitHub core-source synchronization

- `CONFIRMED`: The user explicitly authorized uploading core SG-LL code to
  `https://github.com/Cqhddyb/SG-LL.git` while C1-MS evaluation continues.
- `CONFIRMED`: The local parent Git repository is a different project with
  unrelated user changes. The target remote has existing `LIBERO-plus/` and
  `openvla-oft/` trees at commit
  `3eac3f8f86e7f33eea459c95f5f2dad7e569ba7a`.
- `CONFIRMED`: Prepared the implementation under `SG-LL/`, plus a root README, in an
  isolated clone descended from that remote commit. Do not change existing
  remote files, rewrite history, alter the parent Git repository, or modify
  running experiment sources.
- `CONFIRMED`: The snapshot includes the model, training/configuration/data pipeline, client,
  LIBERO evaluator, relevant tests, offline feature/tokenizer tools, selected
  statistics utilities, dependency lock and research documentation. Exclude
  weights, datasets, feature caches, raw results/logs, local environments,
  private agent memory, and historical machine/queue launch orchestration.
- `CONFIRMED`: In the release copy only, replaced the machine-specific VGGT
  default with `checkpoints/vggt`, the LIBERO evaluator default with
  `../LIBERO-plus`, and local paths in prose with placeholders. Update the
  matching config assertion. Preserve original experiment bytes and record
  original/release hashes and every transformation.
- `CONFIRMED`: Read access works, but HTTPS dry-run push lacks credentials and
  SSH authentication returns `Permission denied (publickey)`. Prepare and test
  a concrete commit before requesting the missing authentication setup. Never
  ask the user to paste a token into chat or store one in the source snapshot.
- `CONFIRMED`: The release copy passes 69 focused CPU tests. One upstream
  ALOHA integration test is excluded from the documented offline command:
  its first attempt reached the external dataset cache and failed on a
  read-only cache lock. No ALOHA data is required for this publication.
  Python AST parsing and Bash syntax pass. Seven inherited VGGT fatal-lint
  findings match the original exactly; no changed Python file has a finding.
  The inherited benchmark config test has the same pre-existing formatting
  finding in both trees; this upload does not reformat research source.
- `PROPOSED`: Commit the reviewed additive snapshot, retain an incremental
  Git bundle locally, and push `HEAD:main` without force after authentication
  is available. Upload status must be confirmed from the remote commit.

Evidence and completion status belong in
`docs/journal/2026-11-11_github_core_source_sync.md` and
`artifacts/source_sync/20261111_github_core/`.
