#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_INDEX = REPO_ROOT / 'advDF' / 'ensemble_test' / 'input_pair_index.xlsx'
PYTHON = sys.executable


def run(command: list[str], expect_success: bool = True) -> subprocess.CompletedProcess:
    print('  $ ' + ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if expect_success and result.returncode != 0:
        raise RuntimeError(f'command failed with {result.returncode}: {command}')
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f'command unexpectedly succeeded: {command}')
    return result


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def create_synthetic_shard(prefix: str, start: int, end: int, pairs: list[tuple[int, int]], loss_type: str) -> Path:
    out = REPO_ROOT / 'outputs' / f'{prefix}_{start}_{end}'
    reset_path(out)
    out.mkdir(parents=True)
    frame = pd.DataFrame({
        'pair_0': [a for a, _ in pairs],
        'pair_1': [b for _, b in pairs],
        'loss': [0.0 for _ in pairs],
    })
    result_name = f'{loss_type}result_loss.xlsx'
    frame.to_excel(out / result_name, index=False, engine='openpyxl')
    manifest = {
        'status': 'complete',
        'pair_index': 'advDF/ensemble_test/input_pair_index.xlsx',
        'pair_start': start,
        'pair_end': end,
        'selected_pairs': end - start,
        'processed_pairs': end - start,
        'missing_pairs': 0,
        'source_model': ['simswap'],
        'lossType': loss_type,
        'testType': 'df',
        'testAttackType': 'fr',
        'relighting': True,
        'total_mask': True,
        'hard_constraint': True,
        'save_every': 1,
        'result_xlsx': result_name,
        'output_files': [],
    }
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return out


def main() -> int:
    print('Tool self-tests:')
    loss_type = 'ensemble_wb_test'
    prefix = 'selftest_air_pairs'
    aggregate_dir = REPO_ROOT / 'outputs' / 'selftest_air_aggregate'
    bad_aggregate_dir = REPO_ROOT / 'outputs' / 'selftest_air_bad_aggregate'
    cleanup_paths = [
        REPO_ROOT / 'outputs' / f'{prefix}_0_2',
        REPO_ROOT / 'outputs' / f'{prefix}_2_4',
        aggregate_dir,
        bad_aggregate_dir,
    ]
    try:
        for path in cleanup_paths:
            reset_path(path)

        pair_df = pd.read_excel(PAIR_INDEX, engine='openpyxl')
        first_four = [(int(a), int(b)) for a, b in pair_df.loc[0:3, ['pair0', 'pair1']].itertuples(index=False, name=None)]
        create_synthetic_shard(prefix, 0, 2, first_four[:2], loss_type)
        create_synthetic_shard(prefix, 2, 4, first_four[2:4], loss_type)

        run([PYTHON, 'tools/aggregate_shards.py', '--shard_size', '2', '--output_prefix', prefix, '--aggregate_dir', str(aggregate_dir), '--total_pairs', '4'])
        run([PYTHON, 'tools/audit_experiment.py', '--shard_size', '2', '--output_prefix', prefix, '--aggregate_dir', str(aggregate_dir), '--total_pairs', '4', '--require_complete'])

        shutil.copytree(aggregate_dir, bad_aggregate_dir)
        bad_xlsx = bad_aggregate_dir / f'{loss_type}aggregate_result_loss.xlsx'
        bad_df = pd.read_excel(bad_xlsx, engine='openpyxl')
        bad_df.loc[[0, 1], ['pair_0', 'pair_1']] = bad_df.loc[[1, 0], ['pair_0', 'pair_1']].to_numpy()
        bad_df.to_excel(bad_xlsx, index=False, engine='openpyxl')
        failed = run([PYTHON, 'tools/audit_experiment.py', '--shard_size', '2', '--output_prefix', prefix, '--aggregate_dir', str(bad_aggregate_dir), '--total_pairs', '4', '--require_complete'], expect_success=False)
        if 'pair mismatch' not in (failed.stdout + failed.stderr):
            raise RuntimeError('corrupted aggregate did not report a pair mismatch')

        run([PYTHON, 'tools/make_public_release.py', '--dest', '/tmp/air_selftest_public_release', '--force'])
        symlinks = list(Path('/tmp/air_selftest_public_release').rglob('*'))
        if any(path.is_symlink() for path in symlinks):
            raise RuntimeError('public release self-test found a remaining symlink')

        print('Tool self-tests passed.')
        return 0
    finally:
        for path in cleanup_paths:
            reset_path(path)


if __name__ == '__main__':
    raise SystemExit(main())
