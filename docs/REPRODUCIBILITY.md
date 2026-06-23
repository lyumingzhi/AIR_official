# Reproducibility Guide

This repository is organized around the paper path for **Transferable Attack against Face Swapping in an Extended Space**. The goal is to keep the release focused on AIR experiments rather than the extra service/application code that existed in `AIR_official`.

## Paper Method Mapping

| Paper component | Code path | Notes |
| --- | --- | --- |
| AIR / RFA relighting attack | `advDF/ensemble_test/attacks.py`, `advDF/ensemble_test/DPR.py` | Optimizes relighting parameters and additive perturbation with DPR. |
| AIA identity attack objective | `advDF/ensemble_test/attacks.py` | Uses an ensemble of face-recognition models. |
| ATI adaptive translation-invariant gradients | `advDF/ensemble_test/attacks.py` | Implemented in the dynamic gradient preprocessing path. |
| 8 FR surrogate models | `scripts/paper_air_args.sh` | ResNet-18/34/50/100 checkpoints across MS1MV3 and Glint360K/CosFace families, shared by full and smoke launchers. |
| 1000-pair experiment driver | `advDF/ensemble_test/import_test_1000.py` | Supports `--dry_run`, `--pair_start`, `--pair_end`, `--max_pairs`, and `--save_every` for staged checks. |

## Expected Data Layout

The full experiment expects aligned face images addressable by the pair indices in `advDF/ensemble_test/input_pair_index.xlsx`. For a portable run, place CelebA-HQ-style images at:

```bash
./data/images
```

For a portable run, place images under:

```text
data/images/
```

or pass another directory with `--dir`.

## Environment

Use the requested Python environment by default. See `docs/TROUBLESHOOTING.md` for runtime issues and `environment.yml` for a conda-style package specification:

```bash
python
```

Before running experiments, the broadest local health check is:

```bash
cd AIR
python tools/release_check.py
```

The individual checks are:

```bash
python tools/check_setup.py
python tools/check_hygiene.py
python tools/smoke_import.py
python tools/check_paper_alignment.py
```

`tools/check_setup.py` verifies packages, linked weights, target-model repos, CUDA, Ninja, and the local `libstdc++` needed by compiled PyTorch extensions. See `docs/ARTIFACTS.md` for the included/external asset policy.

## Fast Validation

Validate the full 1000-pair index against the local image directory without loading models:

```bash
cd AIR
python tools/validate_pair_index.py
```

Argument/data validation for one selected pair without model loading:

```bash
cd AIR
python main.py \
  --source_model simswap \
  --dir ./data/images \
  --output_path ./outputs/dry_run \
  --pair_start 0 \
  --pair_end 1 \
  --dry_run \
  --fail_on_missing_pairs
```

One-pair end-to-end smoke run after installing external assets:

```bash
cd AIR
OUT_DIR=./outputs/smoke_pair1_png CUDA_VISIBLE_DEVICES=0 ./scripts/run_smoke_pair.sh
```

Generated result files are intentionally ignored by git and should not be kept in the public code tree.

## Full Paper-Style Run

```bash
cd AIR
CUDA_VISIBLE_DEVICES=0 ./scripts/run_air_full.sh
```

The default script uses `--source_model simswap`, evaluates transfer against the available local target wrappers, and writes outputs to `./outputs/air_full_simswap`. The full 1000-pair run is compute-heavy; use `--max_pairs N` first when validating a new machine. Add `--fail_on_missing_pairs` to make incomplete datasets fail before model loading instead of silently skipping pairs.

## Sharded 1000-Pair Runs

Plan and audit shards first:

```bash
cd AIR
python tools/shard_status.py --shard_size 100
```

Use zero-based half-open pair ranges when running long experiments in recoverable chunks:

```bash
cd AIR
PAIR_START=0 PAIR_END=100 DATA_DIR=./data/images CUDA_VISIBLE_DEVICES=0 ./scripts/run_air_shard.sh
PAIR_START=100 PAIR_END=200 DATA_DIR=./data/images CUDA_VISIBLE_DEVICES=0 ./scripts/run_air_shard.sh
```

Each shard writes to `outputs/air_simswap_pairs_${PAIR_START}_${PAIR_END}` by default. A completed shard also writes `run_manifest.json`, including pair range, selected/processed/missing counts, result table, and generated output files. Re-run only the failed range if a long job is interrupted. After jobs finish, run `tools/shard_status.py --status --shard_size 100` to list completed and missing ranges; manifest evidence is preferred over older Excel-only checks. When all shards are complete, merge the result tables with:

```bash
python tools/aggregate_shards.py --shard_size 100
```

This writes `outputs/air_simswap_aggregate/aggregate_manifest.json` and `outputs/air_simswap_aggregate/ensemble_wb_testaggregate_result_loss.xlsx`. For small planned subsets or tests, pass `--total_pairs N` to limit the expected pair range.

Before claiming the 1000-pair experiment is complete, audit shard manifests, aggregate rows, and pair order against `input_pair_index.xlsx`:

```bash
python tools/audit_experiment.py --shard_size 100 --require_complete
```

## Known Scope Limits

The current local code covers the core AIR method and the available target wrappers inherited from the local `advDF` codebase: SimSwap, FaceShifter, MegaGAN, AgileGAN, and InfoSwap. The paper also reports DiffFace and DiffSwap target-model experiments; those require complete third-party target wrappers and checkpoints before they can be claimed as reproduced here.

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the current scope gaps and release caveats.
