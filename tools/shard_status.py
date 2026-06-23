#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_INDEX = REPO_ROOT / 'advDF' / 'ensemble_test' / 'input_pair_index.xlsx'


def shard_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + shard_size, total)) for start in range(0, total, shard_size)]


def output_dir(prefix: str, start: int, end: int) -> Path:
    return REPO_ROOT / 'outputs' / f'{prefix}_{start}_{end}'


def load_manifest(path: Path) -> dict | None:
    manifest = path / 'run_manifest.json'
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return None


def shard_done(path: Path, loss_type: str, start: int, end: int) -> tuple[bool, str]:
    manifest = load_manifest(path)
    if manifest is not None:
        expected = end - start
        ok = (
            manifest.get('status') == 'complete'
            and manifest.get('pair_start') == start
            and manifest.get('pair_end') == end
            and manifest.get('processed_pairs') == expected
            and manifest.get('missing_pairs') == 0
        )
        if ok:
            return True, 'manifest'
        return False, 'manifest-mismatch'

    result = path / f'{loss_type}result_loss.xlsx'
    if not result.is_file():
        return False, 'missing'
    try:
        df = pd.read_excel(result, engine='openpyxl')
    except Exception:
        return False, 'bad-xlsx'
    expected = end - start
    ok = {'pair_0', 'pair_1'}.issubset(df.columns) and len(df) == expected
    return ok, 'xlsx' if ok else f'xlsx-rows-{len(df)}'


def main() -> int:
    parser = argparse.ArgumentParser(description='Plan and inspect AIR sharded 1000-pair runs.')
    parser.add_argument('--shard_size', type=int, default=100, help='number of pair rows per shard')
    parser.add_argument('--output_prefix', default='air_simswap_pairs', help='output directory prefix under outputs/')
    parser.add_argument('--loss_type', default='ensemble_wb_test', help='lossType prefix used by result Excel files')
    parser.add_argument(
        '--data_dir',
        default='/home1/mingzhi/advDF/One-Shot-Face-Swapping-on-Megapixels/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img',
        help='DATA_DIR value to print in planned commands',
    )
    parser.add_argument('--cuda_devices', default='0', help='CUDA_VISIBLE_DEVICES value to print in planned commands')
    parser.add_argument('--commands', action='store_true', help='print shell commands for all shards')
    parser.add_argument('--status', action='store_true', help='inspect output directories and summarize done/missing shards')
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise SystemExit('--shard_size must be a positive integer')

    pairs = pd.read_excel(PAIR_INDEX, engine='openpyxl')
    total = len(pairs)
    ranges = shard_ranges(total, args.shard_size)
    print(f'pair_index={PAIR_INDEX.relative_to(REPO_ROOT)} total_pairs={total} shard_size={args.shard_size} shard_count={len(ranges)}')

    if not args.commands and not args.status:
        args.commands = True
        args.status = True

    if args.commands:
        print('\nCommands:')
        for start, end in ranges:
            out = output_dir(args.output_prefix, start, end).relative_to(REPO_ROOT)
            print(
                f'PAIR_START={start} PAIR_END={end} OUT_DIR=./{out} '
                f'DATA_DIR={args.data_dir} CUDA_VISIBLE_DEVICES={args.cuda_devices} ./scripts/run_air_shard.sh'
            )

    if args.status:
        done = []
        missing = []
        print('\nStatus:')
        for start, end in ranges:
            out = output_dir(args.output_prefix, start, end)
            ok, evidence = shard_done(out, args.loss_type, start, end)
            row = f'{start:04d}-{end:04d} {"DONE" if ok else "MISSING"} evidence={evidence} {out.relative_to(REPO_ROOT)}'
            print(row)
            (done if ok else missing).append((start, end))
        print(f'\nSummary: done={len(done)} missing={len(missing)} total={len(ranges)}')
        if missing:
            print('Missing ranges:', ', '.join(f'{start}-{end}' for start, end in missing))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
