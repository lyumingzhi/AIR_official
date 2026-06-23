#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_INDEX = REPO_ROOT / 'advDF' / 'ensemble_test' / 'input_pair_index.xlsx'
DEFAULT_LOCAL_DATA = REPO_ROOT / 'data' / 'images'
REQUIRED_COLUMNS = ('pair0', 'pair1')
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_pairs(pair_index: Path) -> list[tuple[int, int]]:
    if not pair_index.is_file():
        raise SystemExit(f'pair index not found: {pair_index}')
    df = pd.read_excel(pair_index, engine='openpyxl')
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise SystemExit(f'pair index missing required columns: {missing_columns}; found columns={list(df.columns)}')

    pairs: list[tuple[int, int]] = []
    bad_rows: list[str] = []
    for row_number, row in enumerate(df.loc[:, REQUIRED_COLUMNS].itertuples(index=False, name=None), start=2):
        values: list[int] = []
        for value in row:
            try:
                int_value = int(value)
            except Exception:
                bad_rows.append(f'row {row_number}: non-integer value {value!r}')
                continue
            if int_value != value and not (isinstance(value, float) and value.is_integer()):
                bad_rows.append(f'row {row_number}: non-integral value {value!r}')
            values.append(int_value)
        if len(values) == 2:
            pairs.append((values[0], values[1]))
    if bad_rows:
        print('Invalid pair-index rows:')
        for item in bad_rows[:20]:
            print(f'  [FAIL] {item}')
        if len(bad_rows) > 20:
            print(f'  ... {len(bad_rows) - 20} more')
        raise SystemExit(1)
    return pairs


def available_image_names(data_dir: Path) -> set[str]:
    if not data_dir.is_dir():
        raise SystemExit(f'data directory not found: {data_dir}')
    names: set[str] = set()
    for path in data_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            names.add(path.name)
    if not names:
        raise SystemExit(f'no input images found under {data_dir}')
    return names


def validate_pairs(pairs: list[tuple[int, int]], expected_pairs: int | None) -> list[str]:
    errors: list[str] = []
    if expected_pairs is not None and len(pairs) != expected_pairs:
        errors.append(f'expected {expected_pairs} pairs, found {len(pairs)}')
    if not pairs:
        errors.append('pair index is empty')
        return errors

    duplicate_count = len(pairs) - len(set(pairs))
    if duplicate_count:
        errors.append(f'duplicate pair rows: {duplicate_count}')
    self_pairs = [(i, a) for i, (a, b) in enumerate(pairs) if a == b]
    if self_pairs:
        preview = ', '.join(f'row_index={i} image={value}' for i, value in self_pairs[:5])
        errors.append(f'self-pairs found: {len(self_pairs)}; first: {preview}')
    negative_values = [(i, a, b) for i, (a, b) in enumerate(pairs) if a < 0 or b < 0]
    if negative_values:
        preview = ', '.join(f'row_index={i} pair=({a},{b})' for i, a, b in negative_values[:5])
        errors.append(f'negative image ids found: {len(negative_values)}; first: {preview}')
    return errors


def validate_images(pairs: list[tuple[int, int]], image_names: set[str]) -> list[str]:
    missing: list[tuple[int, str]] = []
    for index, (source_id, target_id) in enumerate(pairs):
        for image_id in (source_id, target_id):
            name = f'{image_id}.jpg'
            if name not in image_names:
                missing.append((index, name))
    if not missing:
        return []
    preview = ', '.join(f'row_index={index}:{name}' for index, name in missing[:10])
    return [f'missing referenced .jpg images: {len(missing)}; first: {preview}']


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate AIR input_pair_index.xlsx against the image dataset used by the 1000-pair experiment.')
    parser.add_argument('--pair-index', default=str(DEFAULT_PAIR_INDEX), help='path to input_pair_index.xlsx')
    parser.add_argument('--data-dir', default=str(DEFAULT_LOCAL_DATA), help='directory containing CelebA-HQ image files')
    parser.add_argument('--expected-pairs', type=int, default=1000, help='expected number of pair rows; use 0 to disable')
    parser.add_argument('--skip-data-check', action='store_true', help='validate Excel structure only, without checking image files')
    args = parser.parse_args()

    pair_index = Path(args.pair_index).resolve()
    data_dir = Path(args.data_dir).resolve()
    expected_pairs = None if args.expected_pairs == 0 else args.expected_pairs
    if expected_pairs is not None and expected_pairs <= 0:
        raise SystemExit('--expected-pairs must be positive, or 0 to disable')

    pairs = read_pairs(pair_index)
    errors = validate_pairs(pairs, expected_pairs)
    image_names: set[str] | None = None
    if not args.skip_data_check:
        image_names = available_image_names(data_dir)
        errors.extend(validate_images(pairs, image_names))

    flat_ids = [value for pair in pairs for value in pair]
    print('Pair-index validation:')
    print(f'  pair_index={pair_index}')
    print(f'  rows={len(pairs)}')
    if flat_ids:
        print(f'  unique_images_in_pairs={len(set(flat_ids))}')
        print(f'  min_image_id={min(flat_ids)}')
        print(f'  max_image_id={max(flat_ids)}')
    print(f'  duplicate_pairs={len(pairs) - len(set(pairs))}')
    print(f'  self_pairs={sum(1 for a, b in pairs if a == b)}')
    if image_names is not None:
        print(f'  data_dir={data_dir}')
        print(f'  available_image_files={len(image_names)}')

    if errors:
        print('Pair-index validation failed:')
        for error in errors:
            print(f'  [FAIL] {error}')
        return 1
    print('Pair-index validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
