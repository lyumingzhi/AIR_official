#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home1/mingzhi/anaconda3/envs/py310/bin/python}"
DATA_DIR="${DATA_DIR:-./data/images}"
OUT_DIR="${OUT_DIR:-./outputs/air_full_simswap}"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/paper_air_args.sh"

export PATH="$(dirname "$PYTHON"):$PATH"
for libstdcpp_candidate in /usr/local/cuda-12.9/nsight-systems-2025.1.3/host-linux-x64 /usr/local/cuda-12.9/nsight-compute-2025.2.0/host/linux-desktop-glibc_2_11_3-x64; do
  if [[ -f "$libstdcpp_candidate/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$libstdcpp_candidate:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$libstdcpp_candidate/libstdc++.so.6 ${LD_PRELOAD:-}"
    break
  fi
done

for cuda_candidate in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-11.8; do
  if [[ -n "$cuda_candidate" && -x "$cuda_candidate/bin/nvcc" ]]; then
    export CUDA_HOME="$cuda_candidate"
    export CUDA_PATH="$cuda_candidate"
    export PATH="$cuda_candidate/bin:$PATH"
    break
  fi
done

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" main.py \
  --output_path "$OUT_DIR" \
  --dir "$DATA_DIR" \
  "${AIR_PAPER_ARGS[@]}" \
  "$@"
