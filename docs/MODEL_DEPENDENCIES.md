# External Model Dependencies

This repository does not vendor large checkpoints or full third-party face-swapping repositories. Install or link them at the placeholder paths shown below, then run `python tools/check_setup.py` from the repository root.

## Face Recognition Surrogates

- InsightFace / ArcFace backbones: https://github.com/deepinsight/insightface
  - Expected checkpoint root: `advDF/insightface/recognition/arcface_torch/checkpoint/`
  - The paper configuration uses eight ResNet-18/34/50/100 ArcFace/CosFace checkpoints listed in `scripts/paper_air_args.sh`.

- Legacy ArcFace ResNet-18 helper checkpoint:
  - Expected path: `advDF/ensemble_test/model_for_attack/checkpoints/resnet18_110.pth`
  - This is used by inherited helper code; keep it external unless your license permits redistribution.

## Target Face-Swapping Models

- SimSwap: https://github.com/neuralchen/SimSwap
  - Expected path: `advDF/SimSwap`

- FaceShifter:
  - Expected path: `advDF/Faceshifter`
  - FaceShifter did not ship here as a vendored upstream repository; use a compatible implementation/checkpoint matching the wrapper in `advDF/ensemble_test/Faceshifter.py`.

- MegaFS / MegaGAN support:
  - MegaFS paper/project family: https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels
  - StyleGAN2 dependency used by the local wrapper: https://github.com/rosinality/stylegan2-pytorch
  - Expected path: `advDF/Megagan`

- AgileGAN: https://github.com/GuoxianSong/AgileGAN
  - Expected path: `advDF/AgileGAN`

## Relighting And Parsing

- DPR relighting checkpoint:
  - Expected path: `advDF/DPR/trained_model/trained_model_1024_03.t7`
  - The model definition used by AIR/RFA is included under `advDF/DPR/model/`; the trained weight is external.

- Face parsing / BiSeNet-style checkpoint:
  - Expected path: `advDF/ensemble_test/faceParsing/res/cp/79999_iter.pth`
  - The parsing code is included, but the trained checkpoint is external.

## Dataset

- Put aligned input face images under `data/images/`, or pass another image directory with `--dir` / `DATA_DIR`.
- The included `advDF/ensemble_test/input_pair_index.xlsx` expects image filenames like `0.jpg`, `1.jpg`, etc.

## Check After Installing

```bash
python tools/check_setup.py
python tools/check_artifacts.py
```

`check_setup.py` reports missing repositories, checkpoints, CUDA/Ninja availability, and Python package issues.
