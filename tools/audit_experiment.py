#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_INDEX = REPO_ROOT / 'advDF' / 'ensemble_test' / 'input_pair_index.xlsx'


def shard_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + shard_size, total)) for start in range(0, total, shard_size)]


def output_dir(prefix: str, start: int, end: int) -> Path:
    return REPO_ROOT / 'outputs' / f'{prefix}_{start}_{end}'


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f'bad json {path}: {exc}') from exc


def expected_pair_slice(pair_df: pd.DataFrame, start: int, end: int) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in pair_df.loc[start:end - 1, ['pair0', 'pair1']].itertuples(index=False, name=None)]


def validate_result_table(path: Path, expected_pairs: list[tuple[int, int]], start: int) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f'missing result table: {path}']
    try:
        df = pd.read_excel(path, engine='openpyxl')
    except Exception as exc:
        return [f'cannot read result table {path}: {exc}']
    required = {'pair_0', 'pair_1'}
    if not required.issubset(df.columns):
        return [f'result table missing pair_0/pair_1 columns: {path}']
    if len(df) != len(expected_pairs):
        errors.append(f'result table rows={len(df)} expected={len(expected_pairs)}: {path}')
    actual_pairs = [(int(a), int(b)) for a, b in df.loc[:, ['pair_0', 'pair_1']].itertuples(index=False, name=None)]
    for offset, (actual, expected) in enumerate(zip(actual_pairs, expected_pairs)):
        if actual != expected:
            errors.append(f'pair mismatch at global_index={start + offset}: actual={actual} expected={expected}')
            break
    if 'pair_global_index' in df.columns:
        actual_indexes = [int(value) for value in df['pair_global_index'].tolist()]
        expected_indexes = list(range(start, start + len(df)))
        if actual_indexes != expected_indexes:
            errors.append(f'pair_global_index is not contiguous from {start}: {path}')
    return errors


def audit_shards(pair_df: pd.DataFrame, total: int, shard_size: int, output_prefix: str, loss_type: str) -> tuple[list[str], list[str], list[str]]:
    completed: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    for start, end in shard_ranges(total, shard_size):
        shard = output_dir(output_prefix, start, end)
        label = f'{start}-{end}'
        manifest_path = shard / 'run_manifest.json'
        manifest = read_json(manifest_path)
        if manifest is None:
            missing.append(label)
            continue
        expected_count = end - start
        checks = {
            'status': manifest.get('status') == 'complete',
            'pair_start': manifest.get('pair_start') == start,
            'pair_end': manifest.get('pair_end') == end,
            'selected_pairs': manifest.get('selected_pairs') == expected_count,
            'processed_pairs': manifest.get('processed_pairs') == expected_count,
            'missing_pairs': manifest.get('missing_pairs') == 0,
            'pair_index': manifest.get('pair_index') == 'advDF/ensemble_test/input_pair_index.xlsx',
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            errors.append(f'{label}: manifest check failed: {", ".join(failed)}')
            continue
        result_name = manifest.get('result_xlsx') or f'{loss_type}result_loss.xlsx'
        result_path = shard / result_name
        expected_pairs = expected_pair_slice(pair_df, start, end)
        result_errors = validate_result_table(result_path, expected_pairs, start)
        if result_errors:
            errors.extend(f'{label}: {error}' for error in result_errors)
            continue
        completed.append(label)
    return completed, missing, errors


def audit_aggregate(pair_df: pd.DataFrame, total: int, aggregate_dir: Path, loss_type: str, require_complete: bool) -> tuple[str, list[str]]:
    errors: list[str] = []
    manifest_path = aggregate_dir / 'aggregate_manifest.json'
    manifest = read_json(manifest_path)
    if manifest is None:
        return 'missing', []

    status = str(manifest.get('status', 'unknown'))
    common_checks = {
        'pair_index': manifest.get('pair_index') == 'advDF/ensemble_test/input_pair_index.xlsx',
        'loss_type': manifest.get('loss_type') == loss_type,
    }
    failed = [name for name, ok in common_checks.items() if not ok]
    if failed:
        errors.append(f'aggregate manifest check failed: {", ".join(failed)}')
        return status, errors

    if status != 'complete':
        if require_complete:
            errors.append(f'aggregate is not complete: status={status}')
        return status, errors

    complete_checks = {
        'total_pairs': manifest.get('total_pairs') == total,
        'rows': manifest.get('rows') == total,
    }
    failed = [name for name, ok in complete_checks.items() if not ok]
    if failed:
        errors.append(f'aggregate manifest check failed: {", ".join(failed)}')
    xlsx_name = manifest.get('aggregate_xlsx') or f'{loss_type}aggregate_result_loss.xlsx'
    expected_pairs = expected_pair_slice(pair_df, 0, total)
    errors.extend(validate_result_table(aggregate_dir / xlsx_name, expected_pairs, 0))
    return status, errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Audit AIR 1000-pair experiment outputs against manifests and input_pair_index.xlsx.')
    parser.add_argument('--shard_size', type=int, default=100, help='number of pair rows per shard')
    parser.add_argument('--output_prefix', default='air_simswap_pairs', help='shard output prefix under outputs/')
    parser.add_argument('--loss_type', default='ensemble_wb_test', help='lossType prefix used by result Excel files')
    parser.add_argument('--aggregate_dir', default='./outputs/air_simswap_aggregate', help='aggregate output directory')
    parser.add_argument('--total_pairs', type=int, default=None, help='optional pair-count limit for tests or partial planned experiments')
    parser.add_argument('--require_complete', action='store_true', help='fail when any shard or aggregate output is missing/incomplete')
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise SystemExit('--shard_size must be a positive integer')
    pair_df = pd.read_excel(PAIR_INDEX, engine='openpyxl')
    total = len(pair_df) if args.total_pairs is None else args.total_pairs
    if total <= 0 or total > len(pair_df):
        raise SystemExit(f'--total_pairs must be between 1 and {len(pair_df)}')

    aggregate_dir = (REPO_ROOT / args.aggregate_dir).resolve() if not Path(args.aggregate_dir).is_absolute() else Path(args.aggregate_dir)
    completed, missing, errors = audit_shards(pair_df, total, args.shard_size, args.output_prefix, args.loss_type)
    aggregate_status, aggregate_errors = audit_aggregate(pair_df, total, aggregate_dir, args.loss_type, args.require_complete)
    errors.extend(aggregate_errors)

    shard_count = len(shard_ranges(total, args.shard_size))
    print('AIR experiment audit:')
    print(f'  pair_index={PAIR_INDEX.relative_to(REPO_ROOT)}')
    print(f'  total_pairs={total}')
    print(f'  shard_size={args.shard_size}')
    print(f'  shard_count={shard_count}')
    print(f'  completed_shards={len(completed)}')
    print(f'  missing_shards={len(missing)}')
    print(f'  aggregate_status={aggregate_status}')
    if missing:
        print('  missing_ranges=' + ', '.join(missing))

    if errors:
        print('AIR experiment audit failed:')
        for error in errors[:100]:
            print(f'  [FAIL] {error}')
        if len(errors) > 100:
            print(f'  ... {len(errors) - 100} more')
        return 1
    if args.require_complete and (missing or aggregate_status != 'complete'):
        print('AIR experiment audit incomplete:')
        if missing:
            print(f'  [MISSING] shards: {", ".join(missing)}')
        if aggregate_status != 'complete':
            print(f'  [MISSING] aggregate: {aggregate_dir} status={aggregate_status}')
        return 1
    print('AIR experiment audit passed.' if not missing and aggregate_status == 'complete' else 'AIR experiment audit passed for available evidence; experiment is incomplete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
