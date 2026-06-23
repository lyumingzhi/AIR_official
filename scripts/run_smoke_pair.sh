#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home1/mingzhi/anaconda3/envs/py310/bin/python}"
DATA_DIR="${DATA_DIR:-/home1/mingzhi/advDF/One-Shot-Face-Swapping-on-Megapixels/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img}"
OUT_DIR="${OUT_DIR:-./outputs/smoke_pair1}"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/paper_air_args.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" main.py \
  --output_path "$OUT_DIR" \
  --dir "$DATA_DIR" \
  --max_pairs 1 \
  --save_every 1 \
  --fail_on_missing_pairs \
  "${AIR_PAPER_ARGS[@]}" \
  "$@"
