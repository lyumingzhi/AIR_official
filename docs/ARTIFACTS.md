# Artifacts and External Assets

This code tree is intentionally lightweight. It contains the cleaned AIR paper-code path, scripts, and validation tools, but it does not vendor large checkpoints, datasets, or complete third-party face-swapping repositories.

## Included

- `camera_ready_icme2025_DF_defense_main_paper.pdf`: local copy of the uploaded paper.
- `advDF/ensemble_test/`: pruned AIR paper path: `import_test_1000.py`, `attacks.py`, loader modules, active target wrappers, pair-index files, and required model helper subpackages.
- `advDF/DPR/model/`: DPR relighting model definitions required by RFA.
- `pytorch_colors/`: vendored lightweight dependency used by the original code.
- `scripts/`: paper-style launchers and shared AIR arguments.
- `docs/TARGET_MODEL_ADAPTERS.md`: adapter contract for connecting external face-swapping models without vendoring them.
- `tools/`: setup, artifact-boundary, public-release staging, hygiene, release, paper-alignment, shard-status, and shard-aggregation checks.

## Pruned Legacy Material

The original `advDF/ensemble_test` tree contained many historical side experiments, old `import_test_*` entrypoints, metric/evaluation scripts, temporary input/output folders, and obsolete attack variants. These are not part of the cleaned AIR paper-code path and were moved out of the release tree. The local recoverable backup is:

```text
local backup outside the public release tree
```

The public release should not include that backup directory.

## Linked or External on This Machine

The following paths are expected by the code but are treated as external assets in the public release. See `docs/MODEL_DEPENDENCIES.md` for concise upstream links and install locations.

- InsightFace surrogate FR checkpoints under `advDF/insightface/recognition/arcface_torch/checkpoint/`
- Legacy ArcFace ResNet-18 checkpoint under `advDF/ensemble_test/model_for_attack/checkpoints/`
- BiSeNet face parsing checkpoint under `advDF/ensemble_test/faceParsing/res/cp/`
- DPR relighting checkpoint under `advDF/DPR/trained_model/`
- Target model repositories: SimSwap, FaceShifter, MegaGAN, and AgileGAN

Run this to verify the current machine has the expected assets and only the expected external links:

```bash
cd AIR
python tools/check_setup.py
python tools/check_artifacts.py
```

## Not Included

- CelebA-HQ images or any other dataset images.
- Large model checkpoints as regular files.
- Full third-party target-model source trees as copied assets.
- Complete DiffFace and DiffSwap wrappers/checkpoints. These are paper evaluation targets but are not fully wired in the local cleaned code tree.

## Public Release Staging

A development tree may use symlinks so experiments can reuse external assets. Do not publish those symlinks directly. Instead, create a clean staging tree that replaces external symlinks with small placeholder README files and writes `PUBLIC_RELEASE_MANIFEST.json`:

```bash
cd AIR
python tools/make_public_release.py --dest /tmp/AIR_public_release --force
```

The staging validator fails if any symlink, generated output, unexpected model/archive file, or oversized regular file remains in the public tree.

## Output Policy

Generated results belong under `outputs/` and are ignored by git. A clean public-code tree should keep only:

```text
outputs/.gitkeep
```

Run this before packaging or publishing:

```bash
python tools/check_hygiene.py
python tools/check_artifacts.py
python tools/make_public_release.py --dest /tmp/AIR_public_release --force
python tools/release_check.py --skip-setup
```

## Responsible Use

See `docs/RESPONSIBLE_USE.md` for allowed-use boundaries before sharing or running experiments.

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the current scope gaps and release caveats.
