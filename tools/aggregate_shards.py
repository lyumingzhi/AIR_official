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


def read_manifest(path: Path) -> dict | None:
    manifest_path = path / 'run_manifest.json'
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return None


def load_completed_shard(path: Path, start: int, end: int, loss_type: str) -> tuple[pd.DataFrame, dict]:
    manifest = read_manifest(path)
    if manifest is None:
        raise ValueError('missing run_manifest.json')
    expected = end - start
    checks = {
        'status': manifest.get('status') == 'complete',
        'pair_start': manifest.get('pair_start') == start,
        'pair_end': manifest.get('pair_end') == end,
        'processed_pairs': manifest.get('processed_pairs') == expected,
        'missing_pairs': manifest.get('missing_pairs') == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f'manifest check failed: {", ".join(failed)}')

    result_name = manifest.get('result_xlsx') or f'{loss_type}result_loss.xlsx'
    result_path = path / result_name
    if not result_path.is_file():
        raise ValueError(f'missing result table: {result_name}')
    df = pd.read_excel(result_path, engine='openpyxl')
    required = {'pair_0', 'pair_1'}
    if not required.issubset(df.columns):
        raise ValueError('result table missing pair_0/pair_1 columns')
    if len(df) != expected:
        raise ValueError(f'result table has {len(df)} rows; expected {expected}')
    df.insert(0, 'pair_global_index', range(start, end))
    df.insert(1, 'shard_start', start)
    df.insert(2, 'shard_end', end)
    return df, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description='Aggregate completed AIR shard result tables and manifests.')
    parser.add_argument('--shard_size', type=int, default=100, help='number of pair rows per shard')
    parser.add_argument('--output_prefix', default='air_simswap_pairs', help='input shard output prefix under outputs/')
    parser.add_argument('--loss_type', default='ensemble_wb_test', help='lossType prefix used by result Excel files')
    parser.add_argument('--aggregate_dir', default='./outputs/air_simswap_aggregate', help='where to write aggregate outputs')
    parser.add_argument('--allow_missing', action='store_true', help='write partial aggregates even if some shards are missing')
    parser.add_argument('--total_pairs', type=int, default=None, help='optional pair-count limit for tests or partial planned experiments')
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise SystemExit('--shard_size must be a positive integer')

    pair_df = pd.read_excel(PAIR_INDEX, engine='openpyxl')
    total = len(pair_df) if args.total_pairs is None else args.total_pairs
    if total <= 0 or total > len(pair_df):
        raise SystemExit(f'--total_pairs must be between 1 and {len(pair_df)}')
    ranges = shard_ranges(total, args.shard_size)

    frames: list[pd.DataFrame] = []
    manifests: list[dict] = []
    missing: list[str] = []
    errors: list[str] = []

    for start, end in ranges:
        shard_path = output_dir(args.output_prefix, start, end)
        try:
            frame, manifest = load_completed_shard(shard_path, start, end, args.loss_type)
        except Exception as exc:
            message = f'{start}-{end}: {exc}'
            missing.append(f'{start}-{end}')
            errors.append(message)
            continue
        frames.append(frame)
        manifests.append(manifest)

    if (missing or errors) and not args.allow_missing:
        print('Aggregate aborted because some shards are incomplete:')
        for error in errors:
            print(f'  {error}')
        print('Use --allow_missing to write a partial aggregate.')
        return 1

    aggregate_dir = (REPO_ROOT / args.aggregate_dir).resolve() if not Path(args.aggregate_dir).is_absolute() else Path(args.aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=['pair_global_index', 'shard_start', 'shard_end', 'pair_0', 'pair_1', 'loss'])
    combined_path = aggregate_dir / f'{args.loss_type}aggregate_result_loss.xlsx'
    combined.to_excel(combined_path, index=False, engine='openpyxl')

    summary = {
        'status': 'partial' if missing else 'complete',
        'pair_index': str(PAIR_INDEX.relative_to(REPO_ROOT)),
        'total_pairs': total,
        'shard_size': args.shard_size,
        'shard_count': len(ranges),
        'completed_shards': len(manifests),
        'missing_shards': missing,
        'rows': int(len(combined)),
        'aggregate_xlsx': combined_path.name,
        'input_output_prefix': args.output_prefix,
        'loss_type': args.loss_type,
        'shards': manifests,
    }
    summary_path = aggregate_dir / 'aggregate_manifest.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f'Wrote {combined_path.relative_to(REPO_ROOT)} rows={len(combined)}')
    print(f'Wrote {summary_path.relative_to(REPO_ROOT)} status={summary["status"]}')
    if missing:
        print('Missing shards:', ', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
