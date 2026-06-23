#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-./data/images}"
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
