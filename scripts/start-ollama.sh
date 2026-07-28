#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/configure-env.sh"
export OLLAMA_HOST="127.0.0.1:11434"

pidfile="$ML_LAB_HOME/runs/ollama.pid"
logfile="$ML_LAB_HOME/runs/logs/ollama.log"
if [[ -s "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
  echo "Ollama is already running on 127.0.0.1:11434"
  exit 0
fi
rm -f "$pidfile"

if systemctl --user cat ollama.service >/dev/null 2>&1; then
  systemctl --user start ollama.service
  for _ in {1..30}; do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null; then
      systemctl --user show --property MainPID --value ollama.service >"$pidfile"
      echo "Ollama user service started on 127.0.0.1:11434"
      exit 0
    fi
    sleep 1
  done
  echo "Ollama user service did not become healthy; run: journalctl --user -u ollama -n 100" >&2
  exit 5
fi

nohup "$HOME/.local/bin/ollama" serve >"$logfile" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
for _ in {1..30}; do
  if curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null; then
    echo "Ollama started on 127.0.0.1:11434 (PID $pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Ollama exited during startup; inspect $logfile" >&2
    rm -f "$pidfile"
    exit 4
  fi
  sleep 1
done
echo "Ollama did not become healthy; inspect $logfile" >&2
exit 5
