#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

instance=""
timeout=900
while (($#)); do
  case "$1" in
    --instance) instance="${2:?}"; shift 2 ;;
    --timeout) timeout="${2:?}"; shift 2 ;;
    -h|--help) printf 'Usage: %s --instance ID [--timeout SECONDS]\n' "$0"; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
require_instance_id "$instance"
[[ "$timeout" =~ ^[0-9]+$ ]] || die "--timeout must be an integer."
require_vast_auth
require_cmd vastai
start="$(date +%s)"
while true; do
  tmp="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-status.XXXXXX" 2>/dev/null || mktemp)"
  if vastai show instance "$instance" --raw >"$tmp" 2>/dev/null; then
    status="$(python3 - "$tmp" <<'PY'
import json, sys
v=json.load(open(sys.argv[1], encoding='utf-8'))
if isinstance(v, list): v=v[0] if v else {}
if isinstance(v, dict) and isinstance(v.get('instance'), dict): v=v['instance']
print(v.get('actual_status') or v.get('cur_state') or v.get('status') or 'unknown')
PY
)"
  else
    status="query-failed"
  fi
  rm -f -- "$tmp"
  info "Instance $instance status: $status"
  case "$status" in
    running) exit 0 ;;
    error|failed|offline|destroyed) die "Instance entered terminal status: $status" ;;
  esac
  (( $(date +%s) - start < timeout )) || die "Timed out waiting for instance $instance"
  sleep 10
done
