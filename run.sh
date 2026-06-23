#!/usr/bin/env bash
set -euo pipefail

# Default paper-code entry; set PYTHON=/path/to/python to choose an environment.
exec "$(dirname "$0")/scripts/run_air_full.sh" "$@"
