#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"

model="${1:-}"
port="${2:-8000}"
if [[ -z "$model" ]]; then
  echo "Usage: $0 MODEL_ID_OR_PATH [PORT]" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo "Port must be an integer from 1024 through 65535." >&2
  exit 2
fi
available_kb="$(df -Pk "$ML_LAB_HOME" | awk 'NR==2 {print $4}')"
if (( available_kb < 35 * 1024 * 1024 )) && [[ "${ML_LAB_ALLOW_LOW_DISK:-0}" != "1" ]]; then
  echo "Refusing model startup/download: less than 35 GiB free." >&2
  exit 3
fi
if (( available_kb < 50 * 1024 * 1024 )); then
  echo "Warning: less than 50 GiB free before model startup/download." >&2
fi

pidfile="$ML_LAB_HOME/runs/vllm-${port}.pid"
logfile="$ML_LAB_HOME/runs/logs/vllm-${port}.log"
if [[ -s "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
  echo "A vLLM process recorded in $pidfile is already running." >&2
  exit 4
fi

echo "Starting vLLM on http://127.0.0.1:$port"
echo "Model: $model"
echo "GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
echo "Maximum model length: ${VLLM_MAX_MODEL_LEN:-2048}"
echo "Log: $logfile"

"$ML_LAB_HOME/.venv-vllm/bin/vllm" serve "$model" \
  --host 127.0.0.1 \
  --port "$port" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}" \
  >"$logfile" 2>&1 &
child=$!
printf '%s\n' "$child" >"$pidfile"

cleanup() {
  if kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}
trap cleanup INT TERM EXIT
wait "$child"
status=$?
exit "$status"
