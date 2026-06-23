# Troubleshooting

This page records the local issues that were fixed while turning the project into a clean AIR paper-code tree.

## Use the Requested Python Environment

The default local environment is:

```bash
python
```

The launchers default to this interpreter. You can override it with:

```bash
PYTHON=/path/to/python ./scripts/run_air_full.sh
```

For a fresh conda-style environment, `environment.yml` documents the expected Python 3.10 package set.

## Check Everything First

```bash
cd AIR
python tools/release_check.py
```

For a lighter check when large checkpoints, CUDA, or local data are unavailable:

```bash
python tools/release_check.py --skip-setup --skip-data-dry-run
```

## CUDA_HOME Points to an Old CUDA

The original environment may expose a stale CUDA path such as `/usr/local/cuda-9.2`. `main.py` and the shell launchers now choose an available local CUDA toolkit with `nvcc`, preferring paths such as `/usr/local/cuda` and `/usr/local/cuda-12.9`.

If compiled extensions fail, run:

```bash
python tools/check_setup.py
```

and inspect the `CUDA toolkit nvcc` section.

## MegaGAN / PyTorch C++ Extension Failures

MegaGAN can compile PyTorch extensions and therefore needs:

- `ninja` available on `PATH`
- a valid CUDA toolkit with `nvcc`
- a recent enough `libstdc++.so.6`

The launchers prepend the py310 environment `bin` directory to `PATH`, set a valid CUDA toolkit when available, and preload a newer `libstdc++` from the local CUDA 12.9 installation if present.

## GLIBCXX Symbol Errors

If an extension fails with a missing `GLIBCXX_*` symbol, run:

```bash
python tools/check_setup.py
```

The `libstdc++ for compiled extensions` section should list a candidate from the local CUDA installation. The launchers and `main.py` use this path automatically when present.

## Missing Checkpoints or External Repositories

Large checkpoints and target-model repositories are intentionally external to this cleaned code tree. Missing assets are reported by:

```bash
python tools/check_setup.py
```

See `docs/ARTIFACTS.md` for the included/external asset policy.

## Dataset or Pair Index Problems

Use `--dry_run` before loading models:

```bash
python main.py \
  --source_model simswap \
  --dir ./data/images \
  --output_path ./outputs/dry_run \
  --pair_start 0 \
  --pair_end 1 \
  --dry_run \
  --fail_on_missing_pairs
```

This reports selected and available pairs without loading attack or target models.

## Generated Outputs Make Hygiene Fail

Generated files belong under `outputs/` and are ignored by git. Before publishing, keep only:

```text
outputs/.gitkeep
```

Then run:

```bash
python tools/check_hygiene.py
```
