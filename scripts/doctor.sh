#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
exec "$ML_LAB_HOME/.venv-train/bin/python" -m ml_lab.doctor "$@"
