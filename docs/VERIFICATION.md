# Verification

This file records local verification for the AIR paper-code deployment.

## Environment

- Python: `/home1/mingzhi/anaconda3/envs/py310/bin/python`
- Main preflight: `python tools/check_setup.py`
- CUDA: available on this machine
- Target model repositories and checkpoints are linked from `/home1/mingzhi/advDF`

## Checks Passed

```bash
cd /home1/mingzhi/AIR
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_setup.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/smoke_import.py
/home1/mingzhi/anaconda3/envs/py310/bin/python main.py --help
```

## One-Pair Smoke Experiment

Command:

```bash
cd /home1/mingzhi/AIR
OUT_DIR=./outputs/smoke_pair1_png CUDA_VISIBLE_DEVICES=0 ./scripts/run_smoke_pair.sh
```

Result:

- Exit code: `0`
- Input pair from `advDF/ensemble_test/input_pair_index.xlsx`: `11413.jpg` and `238.jpg`
- Output table: `outputs/smoke_pair1_png/ensemble_wb_testresult_loss.xlsx`
- Output PNG sheets:
  - `ensemble_wb_test_comparision_result0_faceshifter.png`
  - `ensemble_wb_test_comparision_result0_megagan.png`
  - `ensemble_wb_test_result_admin0_faceshifter.png`
  - `ensemble_wb_test_result_admin0_megagan.png`
  - `ensemble_wb_test_source_img0_faceshifter.png`
  - `ensemble_wb_test_source_img0_megagan.png`
  - `ensemble_wb_test_relightted_0_faceshifter.png`
  - `ensemble_wb_test_relightted_0_megagan.png`

The smoke run exercises the paper path: AIR relighting, additive perturbation, eight InsightFace surrogate FR models, and transfer target wrappers for MegaGAN and FaceShifter when `--source_model simswap` is used.

## Artifact Policy

Smoke artifacts are not committed or kept in the cleaned public-code tree. Re-run `scripts/run_smoke_pair.sh` to regenerate them under `outputs/` when needed.

## Run Manifest

Full and sharded non-dry runs now write `run_manifest.json` under the output directory so shard completion can be audited without inspecting logs.

- `tools/validate_pair_index.py` confirms the 1000-row pair index is unique, has no self-pairs, and all referenced local CelebA-HQ `.jpg` files are available.
- `tools/audit_experiment.py` is wired into release checks to validate available shard/aggregate evidence against the pair index; it reports the full experiment incomplete until required shards and aggregate outputs exist.
- `tools/self_test_tools.py` creates synthetic completed shards to test aggregation/audit success and deliberately corrupts aggregate pair order to confirm the audit fails.
