#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

require_cmd vastai
if [[ ! -s "$VAST_KEY_FILE" ]]; then
  info "Vast.ai authentication: PENDING"
  info "Secure setup: run 'vastai set api-key' interactively, then chmod 600 '$VAST_KEY_FILE'."
  exit 1
fi

mode="$(stat -c '%a' "$VAST_KEY_FILE" 2>/dev/null || true)"
if [[ "$mode" != "600" && "$mode" != "400" ]]; then
  info "Vast.ai authentication: INVALID FILE PERMISSIONS"
  exit 1
fi

status_file="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-auth.XXXXXX" 2>/dev/null || mktemp)"
trap 'rm -f -- "$status_file"' EXIT
if vastai show user --raw >"$status_file" 2>/dev/null; then
  info "Vast.ai authentication: CONFIGURED"
  exit 0
fi
info "Vast.ai authentication: FAILED"
exit 1
