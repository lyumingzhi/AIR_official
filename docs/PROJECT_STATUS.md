# Project Status

This page summarizes the current maturity state of the cleaned AIR paper-code tree.

## Current State

The repository is now a paper-code style implementation for **Transferable Attack against Face Swapping in an Extended Space**. The previous app/backend service surface from `AIR_official` has been removed. The main code path is a pruned AIR experiment stack under `advDF/ensemble_test` with DPR relighting support and paper-style launchers.

## Implemented And Verified

- AIA/ATI/RFA method markers are present in `advDF/ensemble_test/attacks.py`.
- The paper-style 8-surrogate FR ensemble is centralized in `scripts/paper_air_args.sh`.
- The default environment is `a Python 3.10 environment`.
- Required local checkpoints and linked target repositories are checked by `tools/check_setup.py`.
- One-pair smoke execution has been verified locally; outputs are not kept in the clean tree.
- Sharded full-experiment execution is supported through `scripts/run_air_shard.sh`.
- `run_manifest.json` is written by non-dry runs for auditability.
- `tools/shard_status.py` plans and audits shard completion.
- `tools/aggregate_shards.py` merges completed shard result tables and writes `aggregate_manifest.json`.
- `tools/release_check.py` performs the main public-code health check.
- Responsible-use, artifact, reproducibility, troubleshooting, limitation, licensing-status, and citation documents are present.

## Current Gaps

- Full 1000-pair experiment outputs are not included and have not been completed in this clean tree.
- DiffFace and DiffSwap are paper evaluation targets but are not fully wired with local wrappers/checkpoints.
- `CITATION.cff` still contains a placeholder `repository-code` URL.
- A final open-source license has not been selected; see `docs/LICENSING.md`.
- Large checkpoints, datasets, and full third-party target repositories remain external or linked assets.

## Primary Verification Command

```bash
cd AIR
python tools/release_check.py
```

For a lighter non-GPU/non-checkpoint check:

```bash
python tools/release_check.py --skip-setup --skip-data-dry-run
```

## Full Experiment Path

1. Verify assets with `tools/check_setup.py`.
2. Plan shards with `tools/shard_status.py --shard_size 100`.
3. Run shards with `scripts/run_air_shard.sh`.
4. Audit shards with `tools/shard_status.py --status --shard_size 100`.
5. Aggregate completed shards with `tools/aggregate_shards.py --shard_size 100`.

## Release Readiness

This tree is ready as an internal mature paper-code baseline. Public release still requires selecting a license, replacing the citation repository URL, and deciding whether DiffFace/DiffSwap support should be added or left as an explicit scope limitation.


## Pruning Note

Legacy `advDF/ensemble_test` side experiments, obsolete entrypoints, metric scripts, temporary folders, and old attack variants have been moved out of the AIR release tree. A local backup is kept at `local backup outside the public release tree` for recovery, but it is not part of the public paper-code release.
