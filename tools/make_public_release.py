#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    '.git',
    '__pycache__',
    'outputs',
}
SKIP_DIR_PREFIXES = {
    'data/images',
}
SKIP_SUFFIXES = {
    '.pyc',
    '.pyo',
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
ALLOW_HEAVY_FILES = {
    'advDF/ensemble_test/input_pair_index.xlsx',
    'advDF/ensemble_test/test_pair_index.xlsx',
    'advDF/ensemble_test/src_id.xlsx',
    'camera_ready_icme2025_DF_defense_main_paper.pdf',
}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def should_skip_dir(rel_dir: str) -> bool:
    name = rel_dir.rsplit('/', 1)[-1]
    if name in SKIP_DIRS:
        return True
    return any(rel_dir == prefix or rel_dir.startswith(prefix + '/') for prefix in SKIP_DIR_PREFIXES)


def rel_self_and_parents(rel_path: str) -> list[str]:
    values = [rel_path]
    for parent in Path(rel_path).parents:
        value = parent.as_posix()
        if value == '.':
            break
        values.append(value)
    return values


def write_placeholder(path: Path, rel_path: str, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '# External Asset Placeholder\n\n'
        f'This {kind} is required by the AIR experiment code but is not vendored in the public release.\n\n'
        f'- Expected repository path: `{rel_path}`\n\n'
        'Install or link the corresponding third-party repository/checkpoint before running experiments that need it.\n'
    )


def copy_public_tree(dest: Path) -> dict:
    copied_files: list[str] = []
    skipped_symlinks: list[dict[str, str]] = []
    skipped_generated: list[str] = []
    skipped_heavy: list[str] = []

    for src in sorted(REPO_ROOT.rglob('*')):
        rel_path = rel(src)
        if rel_path == dest.name or rel_path.startswith(dest.name + '/'):
            continue
        if any(should_skip_dir(part) for part in rel_self_and_parents(rel_path)):
            if src.is_file() or src.is_symlink():
                skipped_generated.append(rel_path)
            continue

        dst = dest / rel_path
        if src.is_symlink():
            target = src.readlink().as_posix()
            kind = 'directory' if src.resolve(strict=False).is_dir() else 'file'
            skipped_symlinks.append({'path': rel_path, 'kind': kind})
            if kind == 'directory':
                write_placeholder(dst / 'README.md', rel_path, kind)
            else:
                write_placeholder(dst.with_name(dst.name + '.README.md'), rel_path, kind)
            continue

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        if not src.is_file():
            continue

        if src.suffix in SKIP_SUFFIXES:
            skipped_generated.append(rel_path)
            continue

        if src.suffix.lower() in HEAVY_SUFFIXES and rel_path not in ALLOW_HEAVY_FILES:
            skipped_heavy.append(rel_path)
            write_placeholder(dst.with_name(dst.name + '.README.md'), rel_path, str(src), 'file')
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(rel_path)

    data_images = dest / 'data' / 'images'
    data_images.mkdir(parents=True, exist_ok=True)
    (data_images / '.gitkeep').touch()
    outputs = dest / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / '.gitkeep').touch()

    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'AIR working tree',
        'copied_files': copied_files,
        'skipped_symlinks': skipped_symlinks,
        'skipped_generated': skipped_generated,
        'skipped_heavy': skipped_heavy,
    }
    (dest / 'PUBLIC_RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def validate_public_tree(dest: Path, max_regular_file_mb: int) -> list[str]:
    errors: list[str] = []
    max_bytes = max_regular_file_mb * 1024 * 1024
    for path in dest.rglob('*'):
        rel_path = path.relative_to(dest).as_posix()
        if path.is_symlink():
            errors.append(f'symlink remains in public tree: {rel_path}')
            continue
        if path.is_file():
            if path.suffix.lower() in HEAVY_SUFFIXES and rel_path not in ALLOW_HEAVY_FILES:
                errors.append(f'heavy artifact remains in public tree: {rel_path}')
            size = path.stat().st_size
            if size > max_bytes and rel_path not in ALLOW_HEAVY_FILES:
                errors.append(f'file exceeds {max_regular_file_mb} MiB: {rel_path} ({size / 1024 / 1024:.1f} MiB)')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a clean AIR public-release staging directory without local symlink targets or generated outputs.')
    parser.add_argument('--dest', required=True, help='destination directory to create')
    parser.add_argument('--force', action='store_true', help='remove destination first if it already exists')
    parser.add_argument('--max-regular-file-mb', type=int, default=20, help='maximum regular file size outside allowed binary artifacts')
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    if dest == REPO_ROOT or REPO_ROOT in dest.parents:
        raise SystemExit('--dest must be outside the AIR source tree')
    if dest.exists():
        if not args.force:
            raise SystemExit(f'destination already exists: {dest}; pass --force to replace it')
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    manifest = copy_public_tree(dest)
    errors = validate_public_tree(dest, args.max_regular_file_mb)
    print(f'Created public release tree: {dest}')
    print(f"  copied_files={len(manifest['copied_files'])}")
    print(f"  skipped_symlinks={len(manifest['skipped_symlinks'])}")
    print(f"  skipped_heavy={len(manifest['skipped_heavy'])}")
    print(f"  skipped_generated={len(manifest['skipped_generated'])}")
    if errors:
        print('Public release validation failed:')
        for error in errors[:100]:
            print(f'  [FAIL] {error}')
        if len(errors) > 100:
            print(f'  ... {len(errors) - 100} more')
        return 1
    print('Public release validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
