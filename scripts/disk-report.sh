#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"

size_of() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf '%-28s %s\n' "$label" "$(du -sh "$path" 2>/dev/null | awk '{print $1}')"
  else
    printf '%-28s %s\n' "$label" "0"
  fi
}

echo "ML Lab disk report"
size_of "Total lab size" "$ML_LAB_HOME"
size_of "Hugging Face cache" "$HF_HOME"
size_of "Dataset cache" "$HF_DATASETS_CACHE"
size_of "Ollama models" "$OLLAMA_MODELS"
size_of "Base models" "$ML_LAB_HOME/models/base"
size_of "Adapters" "$ML_LAB_HOME/models/adapters"
size_of "Checkpoints" "$ML_LAB_HOME/runs/checkpoints"
size_of "Run logs" "$ML_LAB_HOME/runs/logs"
echo
df -h "$ML_LAB_HOME" | awk 'NR==1 || NR==2 {print}'
echo
echo "Ten largest files under the lab"
find "$ML_LAB_HOME" -xdev -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -nr | head -n 10 \
  | awk -F '\t' '{ cmd="numfmt --to=iec --suffix=B " $1; cmd | getline n; close(cmd); print n "\t" $2 }'
