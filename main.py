#!/usr/bin/env python
"""Convenience entry point for AIR paper-code experiments."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _candidate_libstdcpp_dirs() -> list[Path]:
    return [
        Path('/usr/local/cuda-12.9/nsight-systems-2025.1.3/host-linux-x64'),
        Path('/usr/local/cuda-12.9/nsight-compute-2025.2.0/host/linux-desktop-glibc_2_11_3-x64'),
    ]


def _configure_libstdcpp() -> None:
    if os.environ.get('AIR_LIBSTDCXX_REEXEC') == '1':
        return
    current = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    for candidate in _candidate_libstdcpp_dirs():
        if (candidate / 'libstdc++.so.6').is_file() and str(candidate) not in current:
            libstdcpp = candidate / 'libstdc++.so.6'
            os.environ['LD_LIBRARY_PATH'] = f"{candidate}{os.pathsep}" + os.environ.get('LD_LIBRARY_PATH', '')
            existing_preload = os.environ.get('LD_PRELOAD', '')
            if str(libstdcpp) not in existing_preload.split():
                os.environ['LD_PRELOAD'] = f"{libstdcpp} {existing_preload}".strip()
            os.environ['AIR_LIBSTDCXX_REEXEC'] = '1'
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def _configure_cuda_toolkit() -> None:
    candidates = []
    for key in ('CUDA_HOME', 'CUDA_PATH'):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    candidates.extend([Path('/usr/local/cuda'), Path('/usr/local/cuda-12.9'), Path('/usr/local/cuda-11.8')])

    for candidate in candidates:
        if (candidate / 'bin' / 'nvcc').is_file():
            os.environ['CUDA_HOME'] = str(candidate)
            os.environ['CUDA_PATH'] = str(candidate)
            os.environ['PATH'] = f"{candidate / 'bin'}{os.pathsep}" + os.environ.get('PATH', '')
            return


def _show_help_without_target_imports() -> None:
    from advDF.ensemble_test.options import BaseOptions

    parser_owner = BaseOptions()
    parser_owner.initialize()
    parser_owner.parser.print_help()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    py_bin = Path(sys.executable).resolve().parent
    os.environ['PATH'] = f"{py_bin}{os.pathsep}" + os.environ.get('PATH', '')
    _configure_cuda_toolkit()
    _configure_libstdcpp()

    if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        _show_help_without_target_imports()
        return

    runpy.run_module('advDF.ensemble_test.import_test_1000', run_name='__main__')


if __name__ == '__main__':
    main()
