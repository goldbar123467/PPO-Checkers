#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
export OLLAMA_HOST="127.0.0.1:11434"
model="${OLLAMA_SMOKE_MODEL:-qwen3:0.6b}"

curl -fsS --max-time 5 http://127.0.0.1:11434/api/version >/dev/null
available_kb="$(df -Pk "$ML_LAB_HOME" | awk 'NR==2 {print $4}')"
if (( available_kb < 35 * 1024 * 1024 )) && [[ "${ML_LAB_ALLOW_LOW_DISK:-0}" != "1" ]]; then
  echo "Refusing model pull: less than 35 GiB free." >&2
  exit 3
fi
if ! "$HOME/.local/bin/ollama" list | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
  "$HOME/.local/bin/ollama" pull "$model"
fi

response="$(curl -fsS --max-time 180 http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"prompt\":\"Reply with exactly: GPU READY\",\"stream\":false,\"keep_alive\":\"5m\",\"options\":{\"num_predict\":16,\"temperature\":0}}")"
"$ML_LAB_HOME/.venv-train/bin/python" -c \
  'import json,sys; d=json.load(sys.stdin); assert d.get("response", "").strip(); print("Ollama generation: success")' \
  <<<"$response"

ps_output="$("$HOME/.local/bin/ollama" ps)"
printf '%s\n' "$ps_output"
if ! grep -Eiq '100% GPU|GPU' <<<"$ps_output"; then
  echo "Ollama GPU inference: failed to prove GPU layers were loaded" >&2
  exit 7
fi
echo "Ollama GPU inference: success"
