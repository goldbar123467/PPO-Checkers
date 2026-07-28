#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
pidfile="$ML_LAB_HOME/runs/ollama.pid"
if systemctl --user cat ollama.service >/dev/null 2>&1; then
  systemctl --user stop ollama.service
  rm -f "$pidfile"
  echo "Lab Ollama user service stopped."
  exit 0
fi
if [[ ! -s "$pidfile" ]]; then
  echo "No lab-managed Ollama PID file exists."
  exit 0
fi
pid="$(<"$pidfile")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid"
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
fi
rm -f "$pidfile"
echo "Lab-managed Ollama process stopped."
