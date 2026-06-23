# AIR: Transferable Attack against Face Swapping in an Extended Space

This repository is a cleaned paper-code deployment for **Transferable Attack against Face Swapping in an Extended Space**. It removes the extra `app.py`/`backend.py` service layer from `AIR_official` and keeps the research code path used by the paper: AIA + ATI + RFA for transferable attacks against face swapping.

The uploaded paper is kept at `camera_ready_icme2025_DF_defense_main_paper.pdf` for local reference.

## What Matches The Paper

- **AIA**: ensemble face-recognition identity objective is implemented in `advDF/ensemble_test/attacks.py` through `ensemble_face_recognition_loss`.
- **ATI**: adaptive translation-invariant gradient preprocessing is implemented in `advDF/ensemble_test/attacks.py` through `grad_of_dynamic_preprocess`.
- **RFA/AIR**: relighting optimization via DPR spherical-harmonic parameters is implemented in `advDF/ensemble_test/attacks.py` and `advDF/ensemble_test/DPR.py`.
- **Eight FR surrogates**: the command in `scripts/run_air_full.sh` wires the eight ArcFace/CosFace InsightFace checkpoints described in the paper.
- **Target FS wrappers**: SimSwap, FaceShifter, MegaGAN, AgileGAN, InfoSwap wrappers are retained in `advDF/ensemble_test/*`.
- **1000-pair experiment entry**: `advDF/ensemble_test/import_test_1000.py` is preserved as the main experiment driver.

## Layout

```text
AIR/
  CITATION.cff                   # citation metadata for the associated paper
  environment.yml                 # conda-style Python 3.10 environment spec
  main.py                         # forwards CLI args to import_test_1000
  run.sh                          # default launcher using py310
  scripts/paper_air_args.sh        # shared paper AIR arguments
  scripts/run_air_full.sh          # paper-style AIR command
  scripts/run_air_shard.sh         # recoverable pair-range AIR command
  tools/check_setup.py              # dependency/checkpoint preflight
  tools/check_artifacts.py          # external-link and heavy-artifact boundary check
  tools/make_public_release.py      # clean staging tree for public release
  tools/release_check.py            # one-command public-code health check
  tools/shard_status.py             # plan/check recoverable 1000-pair shards
  tools/aggregate_shards.py         # merge completed shard result tables
  tools/audit_experiment.py         # audit shard/aggregate outputs against pair index
  tools/validate_pair_index.py      # validate 1000-pair index against image data
  tools/check_paper_alignment.py    # paper method/configuration alignment check
  tools/smoke_import.py             # lightweight import check
  tools/self_test_tools.py          # synthetic tests for release/audit helpers
  advDF/ensemble_test/             # pruned AIR attack driver, loaders, wrappers, helpers
  advDF/DPR/model/                 # DPR network definitions needed by RFA
  data/images/                     # input images, e.g. CelebA-HQ images
  outputs/                         # generated results
  docs/PROJECT_STATUS.md           # current maturity and gap summary
  docs/ARTIFACTS.md                # included/external asset policy
  docs/RESPONSIBLE_USE.md          # responsible-use boundary
  docs/RELEASE_CHECKLIST.md        # publishing checklist
  docs/KNOWN_LIMITATIONS.md        # current scope gaps
  docs/LICENSING.md                # license status and release action
  docs/TROUBLESHOOTING.md          # CUDA/checkpoint/runtime troubleshooting
```

Large checkpoints and third-party target-model repos are intentionally not copied into this lightweight code tree. On this machine, target-model directories are linked from `/home1/mingzhi/advDF` so local experiments can reuse the existing assets. The legacy side-experiment files from the original `advDF/ensemble_test` tree were pruned from AIR and kept only in `/home1/mingzhi/AIR_pruned_backup` for recovery.

## Environment

The default environment is the one requested by the user:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python
```

Install missing Python packages only if your environment does not already provide them:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/pip install -r requirements.txt
```

A conda-style environment description is also provided in `environment.yml`; the verified local environment remains `/home1/mingzhi/anaconda3/envs/py310`.

## Required Checkpoints

Place model weights at the paths expected by the scripts, or edit `scripts/run_air_full.sh` and the corresponding loader. The important paths are documented in:

- `advDF/insightface/recognition/arcface_torch/checkpoint/README.md`
- `advDF/ensemble_test/model_for_attack/checkpoints/README.md`
- `advDF/ensemble_test/faceParsing/res/cp/README.md`
- `advDF/DPR/trained_model/README.md`


## Preflight Check

For a one-command public-code health check, run:

```bash
cd /home1/mingzhi/AIR
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/release_check.py
```

Before a full experiment, run:

```bash
cd /home1/mingzhi/AIR
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_setup.py
```

The checker reports missing Python packages, required checkpoints, external target-model repos, the Ninja executable required by MegaGAN C++ extensions, and CUDA availability. For a quick import-only check after installing dependencies:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/smoke_import.py
```

To make sure generated outputs, Python caches, unexpected heavy artifacts, and legacy app/backend files are not present in the public code tree:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_hygiene.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/self_test_tools.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_artifacts.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/validate_pair_index.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_paper_alignment.py
```

## Run

From the repository root:

```bash
cd /home1/mingzhi/AIR
./run.sh
```

To print shard commands and inspect completed shard outputs:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/shard_status.py --shard_size 100
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/aggregate_shards.py --shard_size 100
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/audit_experiment.py --shard_size 100 --require_complete
```

For a recoverable chunk of the full experiment, run a zero-based half-open pair range:

```bash
PAIR_START=0 PAIR_END=100 DATA_DIR=/home1/mingzhi/advDF/One-Shot-Face-Swapping-on-Megapixels/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img CUDA_VISIBLE_DEVICES=0 ./scripts/run_air_shard.sh
```

For a verified one-pair smoke experiment on the local CelebA-HQ tree:

```bash
OUT_DIR=./outputs/smoke_pair1_png CUDA_VISIBLE_DEVICES=0 ./scripts/run_smoke_pair.sh
```

See `docs/TARGET_MODEL_ADAPTERS.md` before adding a new face-swapping model. See `docs/PROJECT_STATUS.md` for the current maturity/gap summary, `docs/VERIFICATION.md` for the current verification record, `docs/REPRODUCIBILITY.md` for the reproducibility checklist, `docs/ARTIFACTS.md` for included/external asset policy and public-release staging, `docs/RESPONSIBLE_USE.md` for responsible-use boundaries, and `docs/RELEASE_CHECKLIST.md` before publishing. See `docs/KNOWN_LIMITATIONS.md` for current scope gaps, `docs/LICENSING.md` for license status, and `docs/TROUBLESHOOTING.md` for CUDA, checkpoint, and extension issues.

For a custom dataset, put images under `data/images` or pass another `--dir` by editing `scripts/run_air_full.sh`. You can validate argument wiring without loading models by adding `--dry_run`, select recoverable pair ranges with `--pair_start A --pair_end B`, cap short experiments with `--max_pairs N`, fail early on incomplete pair data with `--fail_on_missing_pairs`, and control PNG sheet flushing with `--save_every N`. Results are written under `outputs/` and each completed run writes `run_manifest.json` for auditability. Output artifacts are ignored by git; keep only `outputs/.gitkeep` in the public code tree. MegaGAN uses PyTorch C++ extensions, so `scripts/run_air_full.sh` prepends the selected conda environment `bin` directory to `PATH` and chooses a CUDA toolkit with a valid `nvcc` before running. It also prepends and preloads a newer `libstdc++` when available so compiled PyTorch extensions can resolve recent `GLIBCXX` symbols. If `CUDA_HOME` points to a stale CUDA, `main.py` will ignore it and use an available local toolkit.

## Responsible Use

This project is for defensive research and reproducibility. Review `docs/RESPONSIBLE_USE.md` before using or sharing the code.

## Notes

This is now a paper-code style repository, not a web application. The previous Gradio/Flask files from `AIR_official` were intentionally removed. Some target face-swapping models and checkpoints remain external because copying them would make this repository very large and would mix third-party model assets into the public code release.

## Limitations And Licensing

Known limitations are summarized in `docs/KNOWN_LIMITATIONS.md`. No final open-source license has been selected yet; see `docs/LICENSING.md` before public redistribution.

## Citation

Citation metadata is provided in `CITATION.cff`. Replace the placeholder `repository-code` URL after publishing the public repository.
