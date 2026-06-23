#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    'paper PDF': 'camera_ready_icme2025_DF_defense_main_paper.pdf',
    'citation metadata': 'CITATION.cff',
    'environment spec': 'environment.yml',
    'project status': 'docs/PROJECT_STATUS.md',
    'artifact policy': 'docs/ARTIFACTS.md',
    'responsible use policy': 'docs/RESPONSIBLE_USE.md',
    'release checklist': 'docs/RELEASE_CHECKLIST.md',
    'troubleshooting guide': 'docs/TROUBLESHOOTING.md',
    'target model adapter guide': 'docs/TARGET_MODEL_ADAPTERS.md',
    'external model dependency guide': 'docs/MODEL_DEPENDENCIES.md',
    'known limitations': 'docs/KNOWN_LIMITATIONS.md',
    'licensing status': 'docs/LICENSING.md',
    'main experiment driver': 'advDF/ensemble_test/import_test_1000.py',
    'attack implementation': 'advDF/ensemble_test/attacks.py',
    'DPR wrapper': 'advDF/ensemble_test/DPR.py',
    'DPR network definition': 'advDF/DPR/model/defineHourglass_1024_gray_skip_matchFeature.py',
    'shared paper args': 'scripts/paper_air_args.sh',
    'paper full launcher': 'scripts/run_air_full.sh',
    'paper shard launcher': 'scripts/run_air_shard.sh',
}

ATTACK_MARKERS = {
    'AIA ensemble FR objective': 'def ensemble_face_recognition_loss',
    'ATI adaptive TI preprocessing': 'def grad_of_dynamic_preprocess',
    'RFA relighting DPR call': 'DPR(self.device)',
    'RFA SH optimization': 'relighting_net.sh',
    'natural SH distribution option': 'SH_NormalDistribution',
}

EXPECTED_SURROGATE_NAMES = ['r18', 'r18', 'r34', 'r34', 'r50', 'r50', 'r100', 'r100']
EXPECTED_CHECKPOINT_KEYWORDS = [
    'ms1mv3_arcface_r18',
    'glint360k_cosface_r18',
    'ms1mv3_arcface_r34',
    'glint360k_cosface_r34',
    'ms1mv3_arcface_r50',
    'glint360k_cosface_r50',
    'glint360k_cosface_r100',
    'ms1mv3_arcface_r100',
]

AVAILABLE_TARGET_WRAPPERS = ['Simswap.py', 'Faceshifter.py', 'Megagan.py', 'AgileGAN.py', 'InfoSwap.py']
PAPER_TARGETS_REQUIRING_EXTERNAL_WORK = ['DiffFace', 'DiffSwap']


def marker(ok: bool) -> str:
    return 'OK' if ok else 'MISSING'


def check_required_files() -> bool:
    print('Required paper-code files:')
    ok_all = True
    for label, rel in REQUIRED_FILES.items():
        ok = (REPO_ROOT / rel).exists()
        ok_all &= ok
        print(f'  [{marker(ok):7}] {label}: {rel}')
    return ok_all


def check_attack_markers() -> bool:
    print('\nAIR method markers in attacks.py:')
    attack_path = REPO_ROOT / 'advDF/ensemble_test/attacks.py'
    text = attack_path.read_text(errors='ignore') if attack_path.exists() else ''
    ok_all = True
    for label, needle in ATTACK_MARKERS.items():
        ok = needle in text
        ok_all &= ok
        print(f'  [{marker(ok):7}] {label}: {needle}')
    return ok_all


def parse_paper_args() -> tuple[list[str], list[str], str]:
    path = REPO_ROOT / 'scripts/paper_air_args.sh'
    text = path.read_text(errors='ignore') if path.exists() else ''
    names = re.findall(r'--name_of_arcface_insight\s+([^\s]+)', text)
    checkpoints = re.findall(r'--checkpoint_of_arcface_insight\s+([^\s]+)', text)
    return names, checkpoints, text


def check_surrogates() -> bool:
    print('\nPaper surrogate FR ensemble:')
    names, checkpoints, text = parse_paper_args()
    ok_names = names == EXPECTED_SURROGATE_NAMES
    ok_checkpoint_count = len(checkpoints) == 8
    ok_checkpoint_keywords = all(any(keyword in ckpt for ckpt in checkpoints) for keyword in EXPECTED_CHECKPOINT_KEYWORDS)
    required_flags = ['--source_model simswap', '--relighting', '--total_mask', '--hard_constraint', '--lossType ensemble_wb_test', '--testType df', '--testAttackType fr']
    ok_flags = all(flag in text for flag in required_flags)
    print(f'  [{marker(ok_names):7}] surrogate names: {names}')
    print(f'  [{marker(ok_checkpoint_count):7}] checkpoint count: {len(checkpoints)}')
    print(f'  [{marker(ok_checkpoint_keywords):7}] checkpoint families: {", ".join(EXPECTED_CHECKPOINT_KEYWORDS)}')
    print(f'  [{marker(ok_flags):7}] paper AIR flags: {", ".join(required_flags)}')
    return ok_names and ok_checkpoint_count and ok_checkpoint_keywords and ok_flags


def check_targets(strict_targets: bool) -> bool:
    print('\nTarget face-swapping wrappers:')
    wrapper_dir = REPO_ROOT / 'advDF/ensemble_test'
    ok_available = True
    for name in AVAILABLE_TARGET_WRAPPERS:
        ok = (wrapper_dir / name).exists()
        ok_available &= ok
        print(f'  [{marker(ok):7}] available wrapper: {name}')
    for name in PAPER_TARGETS_REQUIRING_EXTERNAL_WORK:
        print(f'  [GAP    ] paper-reported target not fully wired here: {name}')
    return ok_available and (not strict_targets)


def check_removed_app_backend() -> bool:
    print('\nRemoved non-paper app/backend surface:')
    forbidden = []
    for path in REPO_ROOT.rglob('*'):
        rel = path.relative_to(REPO_ROOT)
        if path.name in {'app.py', 'backend.py'} or 'backend' in rel.parts or 'app' in rel.parts:
            forbidden.append(rel)
    ok = not forbidden
    print(f'  [{marker(ok):7}] no app/backend files')
    for rel in forbidden[:20]:
        print(f'    {rel}')
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description='Check AIR code alignment with the uploaded paper-code target.')
    parser.add_argument('--strict-targets', action='store_true', help='fail because DiffFace/DiffSwap wrappers are not fully wired')
    args = parser.parse_args()

    checks = [
        check_required_files(),
        check_attack_markers(),
        check_surrogates(),
        check_targets(args.strict_targets),
        check_removed_app_backend(),
    ]
    print('\nKnown scope note: DiffFace and DiffSwap are paper evaluation targets, but complete local wrappers/checkpoints are not present in this cleaned code tree.')
    if all(checks):
        print('Paper alignment check passed for the implemented AIR paper-code scope.')
        return 0
    print('Paper alignment check found gaps.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
