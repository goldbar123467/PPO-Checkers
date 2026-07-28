#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"

execute=0
case "${1:-}" in
  "") ;;
  --execute) execute=1 ;;
  *) echo "Usage: $0 [--execute]" >&2; exit 2 ;;
esac

lab_real="$(realpath -e "$ML_LAB_HOME")"
expected_real="$(realpath -e "$HOME/ml-lab")"
if [[ "$lab_real" != "$expected_real" || "$lab_real" == "/" || "$lab_real" == "$HOME" ]]; then
  echo "Refusing cleanup: lab path did not resolve exactly to $HOME/ml-lab" >&2
  exit 3
fi

candidates=(
  "$ML_LAB_HOME/cache/temporary"
  "$ML_LAB_HOME/cache/triton"
  "$ML_LAB_HOME/cache/torch"
)

echo "Cleanup candidates (contents only):"
for target in "${candidates[@]}"; do
  target_real="$(realpath -e "$target")"
  case "$target_real" in
    "$lab_real"/cache/*) ;;
    *) echo "Refusing unsafe cleanup target: $target_real" >&2; exit 4 ;;
  esac
  find "$target_real" -mindepth 1 -maxdepth 1 -printf '%p\n'
done

if (( ! execute )); then
  echo "Dry run only. Re-run with --execute to remove the listed cache entries."
  exit 0
fi

for target in "${candidates[@]}"; do
  find "$target" -mindepth 1 -depth -delete
done
echo "Removed only temporary, Triton, and Torch cache contents."
echo "Final adapters, raw datasets, Hugging Face model cache, and checkpoints were preserved."
