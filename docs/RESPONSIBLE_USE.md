# Responsible Use

This repository implements research code for **Transferable Attack against Face Swapping in an Extended Space**. The code is intended for defensive research, reproducibility, benchmarking, and studying protections against subject-agnostic face-swapping systems.

## Intended Use

Use this code only for:

- Reproducing the AIR paper experiments with authorized datasets and models.
- Evaluating robustness and transferability of face-swapping defenses.
- Comparing defensive image protection methods under controlled research settings.
- Auditing models and datasets where you have permission to run experiments.

## Prohibited Use

Do not use this code to:

- Attack, alter, or distribute images of people without proper authorization or consent.
- Bypass platform protections, impersonate people, or facilitate harassment, fraud, or privacy violations.
- Generate or improve deceptive face-swapping content.
- Claim reproduced paper results without reporting the exact target models, datasets, checkpoints, and shard manifests used.

## Reporting and Reproducibility

When publishing results based on this code, report:

- The dataset and image-pair index used.
- The surrogate FR checkpoints and target FS models used.
- The exact command or shard ranges.
- `run_manifest.json` for each shard or full run.
- `aggregate_manifest.json` when aggregating sharded experiments.
- Any missing target-model coverage, especially DiffFace and DiffSwap if they are not wired in your local setup.

## Scope Boundary

This cleaned release focuses on the AIR paper-code path. It intentionally removes web app/backend surfaces and does not vendor datasets, private checkpoints, or complete third-party target-model repositories.

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the current scope gaps and release caveats.
