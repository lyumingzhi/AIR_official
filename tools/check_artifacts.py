#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SYMLINKS = {
    'advDF/AgileGAN',
    'advDF/Faceshifter',
    'advDF/Megagan',
    'advDF/SimSwap',
    'advDF/DPR/trained_model/trained_model_1024_03.t7',
    'advDF/ensemble_test/faceParsing/res/cp/79999_iter.pth',
    'advDF/ensemble_test/model_for_attack/checkpoints/resnet18_110.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r100_fp16_0.1/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r18_fp16_0.1/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r34_fp16_0.1/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r50_fp16_0.1/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r100_fp16/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r18_fp16/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r34_fp16/backbone.pth',
    'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r50_fp16/backbone.pth',
}

EXTERNAL_PREFIX = Path('/home1/mingzhi/advDF')
PRUNE_DIRS = {
    '.git',
    'outputs',
    'data/images',
    'advDF/AgileGAN',
    'advDF/Faceshifter',
    'advDF/Megagan',
    'advDF/SimSwap',
}
HEAVY_SUFFIXES = {
    '.ckpt',
    '.h5',
    '.npz',
    '.onnx',
    '.pkl',
    '.pt',
    '.pth',
    '.run',
    '.tar',
    '.t7',
    '.tgz',
    '.zip',
}
ALLOWED_BINARY_FILES = {
    'advDF/ensemble_test/input_pair_index.xlsx',
    'advDF/ensemble_test/test_pair_index.xlsx',
    'advDF/ensemble_test/src_id.xlsx',
    'camera_ready_icme2025_DF_defense_main_paper.pdf',
}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def is_under_pruned_dir(path: Path) -> bool:
    rel_path = rel(path)
    return any(rel_path == item or rel_path.startswith(item + '/') for item in PRUNE_DIRS)


def iter_files_without_external_links() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob('*'):
        if is_under_pruned_dir(path):
            continue
        if path.is_symlink():
            continue
        if path.is_file():
            files.append(path)
    return files


def check_symlinks() -> list[str]:
    errors: list[str] = []
    actual = {rel(path): path for path in REPO_ROOT.rglob('*') if path.is_symlink()}
    missing = sorted(EXPECTED_SYMLINKS - set(actual))
    unexpected = sorted(set(actual) - EXPECTED_SYMLINKS)
    for item in missing:
        errors.append(f'missing expected symlink: {item}')
    for item in unexpected:
        errors.append(f'unexpected symlink: {item} -> {actual[item].readlink()}')
    for item in sorted(EXPECTED_SYMLINKS & set(actual)):
        path = actual[item]
        target = path.resolve(strict=False)
        if not path.exists():
            errors.append(f'broken symlink: {item} -> {path.readlink()}')
        try:
            target.relative_to(EXTERNAL_PREFIX)
        except ValueError:
            errors.append(f'symlink target outside expected external tree: {item} -> {target}')
    return errors


def check_regular_files(max_regular_file_mb: int) -> list[str]:
    errors: list[str] = []
    max_bytes = max_regular_file_mb * 1024 * 1024
    for path in iter_files_without_external_links():
        rel_path = rel(path)
        if rel_path in ALLOWED_BINARY_FILES:
            continue
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if suffix in HEAVY_SUFFIXES:
            errors.append(f'heavy artifact checked in as regular file: {rel_path}')
        if size > max_bytes:
            errors.append(f'regular file exceeds {max_regular_file_mb} MiB: {rel_path} ({size / 1024 / 1024:.1f} MiB)')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Check AIR artifact boundaries for public-code packaging.')
    parser.add_argument('--max-regular-file-mb', type=int, default=20, help='maximum regular file size outside allowed binary artifacts')
    args = parser.parse_args()
    if args.max_regular_file_mb <= 0:
        raise SystemExit('--max-regular-file-mb must be positive')

    errors = check_symlinks() + check_regular_files(args.max_regular_file_mb)
    print('Artifact boundary check:')
    print(f'  expected_symlinks={len(EXPECTED_SYMLINKS)}')
    print(f'  max_regular_file_mb={args.max_regular_file_mb}')
    if errors:
        print('Artifact boundary check failed:')
        for error in errors[:100]:
            print(f'  [FAIL] {error}')
        if len(errors) > 100:
            print(f'  ... {len(errors) - 100} more')
        return 1
    print('Artifact boundary check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
