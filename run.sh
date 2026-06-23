#!/usr/bin/env bash
set -euo pipefail

# Default paper-code entry using the requested Python 3.10 conda environment.
exec "$(dirname "$0")/scripts/run_air_full.sh" "$@"
