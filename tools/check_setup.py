#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PYTHON_PACKAGES = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('PIL', 'Pillow'),
    ('pandas', 'pandas'),
    ('openpyxl', 'openpyxl'),
    ('skimage', 'scikit-image'),
    ('lpips', 'lpips'),
    ('facenet_pytorch', 'facenet-pytorch'),
    ('ninja', 'ninja'),
]

CORE_FILES = [
    ('paper PDF', 'camera_ready_icme2025_DF_defense_main_paper.pdf'),
    ('environment spec', 'environment.yml'),
    ('main experiment entry', 'advDF/ensemble_test/import_test_1000.py'),
    ('AIR attack implementation', 'advDF/ensemble_test/attacks.py'),
    ('DPR relighting network definition', 'advDF/DPR/model/defineHourglass_1024_gray_skip_matchFeature.py'),
    ('DPR relighting network helper', 'advDF/DPR/model/defineHourglass_512_gray_skip.py'),
    ('vendored pytorch_colors', 'pytorch_colors/__init__.py'),
    ('shared paper AIR args', 'scripts/paper_air_args.sh'),
    ('full paper launcher', 'scripts/run_air_full.sh'),
    ('one-pair smoke launcher', 'scripts/run_smoke_pair.sh'),
    ('sharded paper launcher', 'scripts/run_air_shard.sh'),
    ('release check tool', 'tools/release_check.py'),
    ('shard status tool', 'tools/shard_status.py'),
    ('shard aggregation tool', 'tools/aggregate_shards.py'),
    ('paper alignment checker', 'tools/check_paper_alignment.py'),
    ('troubleshooting guide', 'docs/TROUBLESHOOTING.md'),
]

REQUIRED_CHECKPOINTS = [
    ('legacy ArcFace ResNet-18', 'advDF/ensemble_test/model_for_attack/checkpoints/resnet18_110.pth'),
    ('BiSeNet face parsing', 'advDF/ensemble_test/faceParsing/res/cp/79999_iter.pth'),
    ('DPR relighting', 'advDF/DPR/trained_model/trained_model_1024_03.t7'),
    ('InsightFace r18 MS1MV3', 'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r18_fp16/backbone.pth'),
    ('InsightFace r18 Glint360K', 'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r18_fp16_0.1/backbone.pth'),
    ('InsightFace r34 MS1MV3', 'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r34_fp16/backbone.pth'),
    ('InsightFace r34 Glint360K', 'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r34_fp16_0.1/backbone.pth'),
    ('InsightFace r50 MS1MV3', 'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r50_fp16/backbone.pth'),
    ('InsightFace r50 Glint360K', 'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r50_fp16_0.1/backbone.pth'),
    ('InsightFace r100 Glint360K', 'advDF/insightface/recognition/arcface_torch/checkpoint/glint360k_cosface_r100_fp16_0.1/backbone.pth'),
    ('InsightFace r100 MS1MV3', 'advDF/insightface/recognition/arcface_torch/checkpoint/ms1mv3_arcface_r100_fp16/backbone.pth'),
]

OPTIONAL_EXTERNAL_DIRS = [
    ('SimSwap target model repo', 'advDF/SimSwap'),
    ('FaceShifter target model repo', 'advDF/Faceshifter'),
    ('MegaGAN target model repo', 'advDF/Megagan'),
    ('AgileGAN target model repo', 'advDF/AgileGAN'),
]


def marker(ok: bool) -> str:
    return 'OK' if ok else 'MISSING'


def check_packages() -> bool:
    print('Python packages:')
    all_ok = True
    for module, package in PYTHON_PACKAGES:
        ok = importlib.util.find_spec(module) is not None
        all_ok &= ok
        print(f'  [{marker(ok):7}] {package} ({module})')
    return all_ok


def check_files(title: str, items: list[tuple[str, str]], required: bool = True) -> bool:
    print(f'{title}:')
    all_ok = True
    for label, rel in items:
        path = REPO_ROOT / rel
        ok = path.exists()
        if required:
            all_ok &= ok
        suffix = ''
        if path.is_symlink():
            suffix = f' -> {os.readlink(path)}'
        print(f'  [{marker(ok):7}] {label}: {rel}{suffix}')
    return all_ok



def check_ninja_executable() -> bool:
    env_bin = Path(sys.executable).resolve().parent
    candidates = [env_bin / 'ninja']
    found = shutil.which('ninja')
    if found:
        candidates.append(Path(found))
    ok = any(path.is_file() and os.access(path, os.X_OK) for path in candidates)
    print(f'Ninja executable: {marker(ok)}')
    for path in candidates:
        if path.exists():
            print(f'  candidate: {path}')
    return ok



def check_libstdcpp() -> bool:
    candidates = [
        Path('/usr/local/cuda-12.9/nsight-systems-2025.1.3/host-linux-x64/libstdc++.so.6'),
        Path('/usr/local/cuda-12.9/nsight-compute-2025.2.0/host/linux-desktop-glibc_2_11_3-x64/libstdc++.so.6'),
    ]
    ok = any(path.is_file() for path in candidates)
    print(f'libstdc++ for compiled extensions: {marker(ok)}')
    for path in candidates:
        if path.is_file():
            print(f'  candidate: {path}')
    return ok

def check_cuda_toolkit() -> bool:
    candidates = []
    for key in ('CUDA_HOME', 'CUDA_PATH'):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    candidates.extend([Path('/usr/local/cuda'), Path('/usr/local/cuda-12.9'), Path('/usr/local/cuda-11.8')])

    seen = set()
    valid = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        nvcc = candidate / 'bin' / 'nvcc'
        if nvcc.is_file():
            valid.append(candidate)

    ok = bool(valid)
    print(f'CUDA toolkit nvcc: {marker(ok)}')
    for candidate in valid:
        print(f'  candidate: {candidate / "bin" / "nvcc"}')
    if os.environ.get('CUDA_HOME') and not (Path(os.environ['CUDA_HOME']) / 'bin' / 'nvcc').is_file():
        print(f'  invalid CUDA_HOME ignored by main.py: {os.environ["CUDA_HOME"]}')
    return ok

def check_cuda() -> bool:
    try:
        import torch
    except Exception as exc:
        print(f'CUDA: unable to import torch: {exc}')
        return False
    ok = torch.cuda.is_available()
    print(f'CUDA: {marker(ok)}')
    if ok:
        print(f'  devices: {torch.cuda.device_count()}')
        print(f'  current: {torch.cuda.get_device_name(0)}')
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description='Check AIR paper-code environment and assets.')
    parser.add_argument('--strict', action='store_true', help='Fail if optional target-model directories are missing.')
    args = parser.parse_args()

    print(f'Repository: {REPO_ROOT}')
    print(f'Python: {sys.executable}')
    print()

    packages_ok = check_packages()
    print()
    core_ok = check_files('Core files', CORE_FILES, required=True)
    print()
    checkpoints_ok = check_files('Required checkpoints for full paper AIR run', REQUIRED_CHECKPOINTS, required=True)
    print()
    optional_ok = check_files('External target model repositories', OPTIONAL_EXTERNAL_DIRS, required=args.strict)
    print()
    ninja_ok = check_ninja_executable()
    print()
    libstdcpp_ok = check_libstdcpp()
    print()
    cuda_toolkit_ok = check_cuda_toolkit()
    print()
    cuda_ok = check_cuda()

    ok = packages_ok and core_ok and checkpoints_ok and optional_ok and ninja_ok and libstdcpp_ok and cuda_toolkit_ok and cuda_ok
    print()
    if ok:
        print('Setup check passed for the full paper-style AIR run.')
        return 0
    print('Setup check found missing items. Install dependencies and place/link checkpoints before full experiments.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
