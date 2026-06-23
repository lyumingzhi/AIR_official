# Known Limitations

This file centralizes the current scope boundaries of the cleaned AIR paper-code tree.

## Target Model Coverage

Implemented or retained local target wrappers:

- SimSwap
- FaceShifter
- MegaGAN
- AgileGAN
- InfoSwap

Paper evaluation targets not fully wired in this cleaned local tree:

- DiffFace
- DiffSwap

Do not claim reproduction of DiffFace or DiffSwap results unless complete wrappers, checkpoints, commands, and run manifests are added and verified.

## Experiment Completion

The code supports the full 1000-pair experiment through `scripts/run_air_full.sh` and recoverable sharded runs through `scripts/run_air_shard.sh`, but this cleaned tree does not include completed 1000-pair outputs.

Use:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/shard_status.py --shard_size 100
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/aggregate_shards.py --shard_size 100
```

to audit and aggregate completed shard outputs.

## External Assets

Datasets, large checkpoints, and complete third-party target-model repositories are not vendored as regular files. See `docs/ARTIFACTS.md` and `tools/check_setup.py` for the exact expected local paths.

## Citation Metadata

`CITATION.cff` contains a placeholder `repository-code` URL. Replace it with the public repository URL before publishing.

## License Status

A final open-source license has not been selected in this cleaned tree. See `docs/LICENSING.md`.
