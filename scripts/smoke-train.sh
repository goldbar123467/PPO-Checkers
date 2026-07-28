#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"

available_kb="$(df -Pk "$ML_LAB_HOME" | awk 'NR==2 {print $4}')"
if (( available_kb < 35 * 1024 * 1024 )) && [[ "${ML_LAB_ALLOW_LOW_DISK:-0}" != "1" ]]; then
  echo "Refusing training: less than 35 GiB free. Set ML_LAB_ALLOW_LOW_DISK=1 only after review." >&2
  exit 3
fi
if (( available_kb < 50 * 1024 * 1024 )); then
  echo "Warning: less than 50 GiB free before training." >&2
fi

cd "$ML_LAB_HOME"
exec "$ML_LAB_HOME/.venv-train/bin/python" -m ml_lab.train_sft \
  --config "$ML_LAB_HOME/configs/smoke-train.yaml" "$@"
