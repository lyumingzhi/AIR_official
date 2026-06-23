# Release Checklist

Use this checklist before publishing or packaging the AIR paper-code tree.

## Required Checks

```bash
cd /home1/mingzhi/AIR
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/release_check.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_paper_alignment.py
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/check_hygiene.py
```

For a lighter check when checkpoints/CUDA are unavailable:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/release_check.py --skip-setup --skip-data-dry-run
```

## Files That Should Exist

- `README.md`
- `CITATION.cff`
- `requirements.txt`
- `environment.yml`
- `docs/PROJECT_STATUS.md`
- `docs/ARTIFACTS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/RESPONSIBLE_USE.md`
- `docs/VERIFICATION.md`
- `docs/TROUBLESHOOTING.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/LICENSING.md`
- `tools/check_setup.py`
- `tools/release_check.py`
- `tools/check_paper_alignment.py`

## Files That Should Not Be Published

- Generated files under `outputs/`, except `outputs/.gitkeep`.
- `__pycache__/` directories and `.pyc` files.
- Legacy app/backend entrypoints from `AIR_official`.
- Local datasets and large checkpoints copied as regular files.

## Before Public Release

- Replace the placeholder `repository-code` field in `CITATION.cff` with the public repository URL.
- Choose and add an explicit root-level `LICENSE` file. Update `docs/LICENSING.md` after doing so.
- Confirm whether DiffFace/DiffSwap wrappers and checkpoints are included. If not, keep the scope limitation in `docs/ARTIFACTS.md` and `docs/REPRODUCIBILITY.md`.
- Run a one-pair smoke experiment on the release machine if GPU/checkpoints are available.
- Record any full or sharded experiment outputs with `run_manifest.json` and `aggregate_manifest.json`.
