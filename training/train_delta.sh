#!/usr/bin/env bash
set -euo pipefail

# Use an explicit override when desired; otherwise preserve the active shell's
# Python resolution. All project-owned paths are derived inside train_delta.py.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/train_delta.py" "$@"
