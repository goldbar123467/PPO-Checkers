#!/usr/bin/env bash

# Shared safety and connection helpers for the Vast.ai wrappers.
set -Eeuo pipefail

VAST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ML_LAB_HOME="$(cd -- "$VAST_SCRIPT_DIR/../.." && pwd -P)"
export ML_LAB_HOME

if [[ -f "$ML_LAB_HOME/scripts/configure-env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ML_LAB_HOME/scripts/configure-env.sh"
fi

VAST_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/vastai"
VAST_KEY_FILE="$VAST_CONFIG_HOME/vast_api_key"
VAST_SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_ed25519_vast_ml}"
VAST_KNOWN_HOSTS="${VAST_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_vast_ml}"
VAST_PROFILE_DIR="$ML_LAB_HOME/cloud/vast/profiles"
VAST_MANIFEST_DIR="$ML_LAB_HOME/runs/remote/manifests"
VAST_RECOVERED_DIR="$ML_LAB_HOME/runs/remote/recovered"
VAST_LOG_DIR="$ML_LAB_HOME/runs/remote/logs"

mkdir -p -- "$VAST_MANIFEST_DIR" "$VAST_RECOVERED_DIR" "$VAST_LOG_DIR"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '%s\n' "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_vast_auth() {
  [[ -s "$VAST_KEY_FILE" ]] || die "Vast.ai authentication is not configured. Run: vastai set api-key (interactively; do not place a key in shell history)."
  local mode
  mode="$(stat -c '%a' "$VAST_KEY_FILE" 2>/dev/null || true)"
  [[ "$mode" == "600" || "$mode" == "400" ]] || die "Vast.ai credential file permissions must be 600 or 400 (currently ${mode:-unknown})."
}

require_instance_id() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] || die "Instance ID must be numeric."
}

require_positive_number() {
  local label="$1" value="$2"
  python3 - "$label" "$value" <<'PY'
import sys
label, raw = sys.argv[1:]
try:
    value = float(raw)
except ValueError:
    raise SystemExit(f"ERROR: {label} must be numeric")
if value <= 0:
    raise SystemExit(f"ERROR: {label} must be greater than zero")
PY
}

resolve_profile() {
  local requested="$1"
  if [[ -f "$requested" ]]; then
    PROFILE_PATH="$(realpath -- "$requested")"
  elif [[ -f "$VAST_PROFILE_DIR/$requested" ]]; then
    PROFILE_PATH="$VAST_PROFILE_DIR/$requested"
  elif [[ -f "$VAST_PROFILE_DIR/$requested.yaml" ]]; then
    PROFILE_PATH="$VAST_PROFILE_DIR/$requested.yaml"
  else
    die "Profile not found: $requested"
  fi
  [[ "$PROFILE_PATH" == "$VAST_PROFILE_DIR/"* ]] || die "Profiles must be under $VAST_PROFILE_DIR"
  export PROFILE_PATH
}

yaml_scalar() {
  local key="$1" file="${2:-$PROFILE_PATH}"
  python3 - "$key" "$file" <<'PY'
import re, sys
key, path = sys.argv[1:]
pattern = re.compile(r"^" + re.escape(key) + r"\s*:\s*(.*?)\s*$")
for line in open(path, encoding="utf-8"):
    match = pattern.match(line)
    if match:
        value = match.group(1).split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        print(value)
        break
PY
}

manifest_append() {
  local run_id="$1" event="$2" payload="${3:-{}}"
  python3 "$VAST_SCRIPT_DIR/_manifest.py" append \
    --manifest "$VAST_MANIFEST_DIR/$run_id.jsonl" \
    --run-id "$run_id" --event "$event" --payload "$payload"
}

manifest_for_instance() {
  python3 "$VAST_SCRIPT_DIR/_manifest.py" find \
    --directory "$VAST_MANIFEST_DIR" --instance-id "$1"
}

manifest_run_id() {
  basename -- "$1" .jsonl
}

manifest_has_event() {
  python3 "$VAST_SCRIPT_DIR/_manifest.py" has --manifest "$1" --event "$2"
}

ssh_connection() {
  local instance_id="$1" raw url authority
  require_instance_id "$instance_id"
  require_vast_auth
  require_cmd vastai
  raw="$(vastai ssh-url "$instance_id" 2>/dev/null)" || die "Could not obtain SSH URL for instance $instance_id"
  url="$(grep -Eo 'ssh://[^[:space:]]+' <<<"$raw" | tail -n1 || true)"
  if [[ -n "$url" ]]; then
    authority="${url#ssh://}"
    SSH_TARGET="${authority%:*}"
    SSH_PORT="${authority##*:}"
  else
    SSH_PORT="$(sed -nE 's/.*ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+([^[:space:]]+).*/\1/p' <<<"$raw" | tail -n1)"
    SSH_TARGET="$(sed -nE 's/.*ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+([^[:space:]]+).*/\2/p' <<<"$raw" | tail -n1)"
  fi
  [[ "$SSH_PORT" =~ ^[0-9]+$ && "$SSH_TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$ ]] || die "Vast.ai returned an unrecognized SSH URL."
  [[ -f "$VAST_SSH_KEY" ]] || die "Dedicated SSH private key not found at $VAST_SSH_KEY"
  local key_mode
  key_mode="$(stat -c '%a' "$VAST_SSH_KEY")"
  [[ "$key_mode" == "600" || "$key_mode" == "400" ]] || die "SSH private key permissions must be 600 or 400."
  mkdir -p -- "$(dirname -- "$VAST_KNOWN_HOSTS")"
  touch -- "$VAST_KNOWN_HOSTS"
  chmod 600 -- "$VAST_KNOWN_HOSTS"
  export SSH_TARGET SSH_PORT
}

remote_shell() {
  local instance_id="$1" command="$2"
  ssh_connection "$instance_id"
  ssh -i "$VAST_SSH_KEY" -p "$SSH_PORT" \
    -o BatchMode=yes -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$VAST_KNOWN_HOSTS" \
    -o ConnectTimeout=20 -- "$SSH_TARGET" "bash -lc $(printf '%q' "$command")"
}

rsync_ssh_command() {
  printf 'ssh -i %q -p %q -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=%q' \
    "$VAST_SSH_KEY" "$SSH_PORT" "$VAST_KNOWN_HOSTS"
}

confirm_execute() {
  local execute="$1" accept="$2"
  [[ "$execute" == "true" ]] || {
    info "DRY RUN: no paid resource was created or changed. Add --execute only after reviewing this plan."
    return 1
  }
  [[ "$accept" == "true" ]] || die "--execute also requires --accept-cost."
  return 0
}
