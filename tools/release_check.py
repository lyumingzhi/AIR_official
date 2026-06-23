#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DATA = REPO_ROOT / 'data' / 'images'


def run_step(label: str, command: list[str], env: dict[str, str] | None = None) -> bool:
    print(f'\n==> {label}', flush=True)
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f'FAILED: {label} exited with {result.returncode}', flush=True)
        return False
    print(f'OK: {label}', flush=True)
    return True


def clean_generated_state() -> None:
    outputs = REPO_ROOT / 'outputs'
    outputs.mkdir(exist_ok=True)
    (outputs / '.gitkeep').touch()
    for child in outputs.iterdir():
        if child.name == '.gitkeep':
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for cache in REPO_ROOT.rglob('__pycache__'):
        shutil.rmtree(cache, ignore_errors=True)
    for pyc in REPO_ROOT.rglob('*.pyc'):
        pyc.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run AIR public-code release checks.')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter to use for checks')
    parser.add_argument('--data-dir', default=str(DEFAULT_LOCAL_DATA), help='Image directory for dry-run pair validation')
    parser.add_argument('--skip-setup', action='store_true', help='Skip dependency/checkpoint/CUDA preflight')
    parser.add_argument('--skip-data-dry-run', action='store_true', help='Skip local dataset dry-run validation')
    args = parser.parse_args()

    python = str(Path(args.python).resolve())
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_setup:
        steps.append(('setup preflight', [python, 'tools/check_setup.py']))
    steps.extend([
        ('script syntax: paper args', ['bash', '-n', 'scripts/paper_air_args.sh']),
        ('script syntax: full launcher', ['bash', '-n', 'scripts/run_air_full.sh']),
        ('script syntax: smoke launcher', ['bash', '-n', 'scripts/run_smoke_pair.sh']),
        ('script syntax: shard launcher', ['bash', '-n', 'scripts/run_air_shard.sh']),
        ('main help', [python, 'main.py', '--help']),
        ('smoke imports', [python, 'tools/smoke_import.py']),
        ('tool self-tests', [python, 'tools/self_test_tools.py']),
        ('paper alignment', [python, 'tools/check_paper_alignment.py']),
        ('artifact boundaries', [python, 'tools/check_artifacts.py']),
        ('public release staging', [python, 'tools/make_public_release.py', '--dest', '/tmp/air_public_release_check', '--force']),
        ('repository hygiene', [python, 'tools/check_hygiene.py']),
        ('shard plan/status', [python, 'tools/shard_status.py', '--shard_size', '250']),
        ('shard aggregation', [python, 'tools/aggregate_shards.py', '--shard_size', '250', '--allow_missing', '--aggregate_dir', './outputs/release_check_aggregate']),
        ('experiment audit', [python, 'tools/audit_experiment.py', '--shard_size', '250', '--aggregate_dir', './outputs/release_check_aggregate']),
    ])
    if not args.skip_data_dry_run:
        steps.append(('pair-index data validation', [python, 'tools/validate_pair_index.py', '--data-dir', args.data_dir]))
        steps.append((
            'local data dry-run',
            [
                python,
                'main.py',
                '--source_model', 'simswap',
                '--dir', args.data_dir,
                '--output_path', './outputs/dry_run',
                '--pair_start', '0',
                '--pair_end', '1',
                '--dry_run',
                '--fail_on_missing_pairs',
            ],
        ))

    clean_generated_state()
    all_ok = True
    for label, command in steps:
        all_ok = run_step(label, command, env=env) and all_ok
    clean_generated_state()

    all_ok = run_step('final repository hygiene', [python, 'tools/check_hygiene.py'], env=env) and all_ok
    if all_ok:
        print('\nRelease check passed.', flush=True)
        return 0
    print('\nRelease check failed.', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
