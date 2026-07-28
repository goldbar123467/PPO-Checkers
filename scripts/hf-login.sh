#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"

cat <<'NOTICE'
Hugging Face authentication uses the interactive browser/device flow.
Never place a token in AGENTS.md, README files, tracked configuration, shell
history, source code, or a command-line argument.
NOTICE

"$ML_LAB_HOME/.venv-train/bin/hf" auth login
"$ML_LAB_HOME/.venv-train/bin/hf" auth whoami
