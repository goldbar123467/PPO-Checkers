#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

execute=false
accept_change=false
key="$VAST_SSH_KEY"
while (($#)); do
  case "$1" in
    --key) key="${2:?missing key path}"; shift 2 ;;
    --execute) execute=true; shift ;;
    --accept-account-change) accept_change=true; shift ;;
    -h|--help)
      printf 'Usage: %s [--key PATH] [--execute --accept-account-change]\n' "$0"
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

if [[ ! -f "$key.pub" ]]; then
  info "Dedicated Vast SSH identity: PENDING"
  info "Generate it without overwriting another key:"
  info "  ssh-keygen -t ed25519 -f '$HOME/.ssh/id_ed25519_vast_ml' -C 'vast-ml-lab'"
  info "Then rerun this script. The private key must never enter the repository."
  exit 1
fi
[[ -f "$key" ]] || die "Public key exists but private key is missing: $key"
mode="$(stat -c '%a' "$key")"
[[ "$mode" == "600" || "$mode" == "400" ]] || die "Set the private key mode to 600 before continuing."
ssh-keygen -l -f "$key.pub"
info "DRY RUN: would register the public key file $key.pub with the Vast.ai account."
if [[ "$execute" != true ]]; then
  info "No account change made. Add --execute --accept-account-change after explicit authorization."
  exit 0
fi
[[ "$accept_change" == true ]] || die "--execute requires --accept-account-change."
require_vast_auth
vastai create ssh-key "$key.pub" -y >/dev/null
info "Dedicated Vast SSH public key registration succeeded."
