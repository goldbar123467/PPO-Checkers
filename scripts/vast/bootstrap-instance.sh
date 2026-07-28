#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

instance=""
execute=false
while (($#)); do
  case "$1" in
    --instance) instance="${2:?}"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) printf 'Usage: %s --instance ID [--execute]\n' "$0"; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
require_instance_id "$instance"
info "Bootstrap target: instance $instance"
info "Installs OS tools and uv only; it does not transfer credentials, environments, models, or data."
if [[ "$execute" != true ]]; then
  info "DRY RUN: add --execute to bootstrap this existing instance."
  exit 0
fi
command='set -Eeuo pipefail; export DEBIAN_FRONTEND=noninteractive; mkdir -p /workspace/ml-lab /workspace/data /workspace/models /workspace/runs /workspace/cache/temporary; if command -v apt-get >/dev/null; then apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl git htop jq openssh-client python3.12 python3.12-dev python3.12-venv rsync tmux && rm -rf /var/lib/apt/lists/*; fi; if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh; fi; /root/.local/bin/uv --version || uv --version; python3.12 --version; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
remote_shell "$instance" "$command"
manifest="$(manifest_for_instance "$instance")"
manifest_append "$(manifest_run_id "$manifest")" "instance_bootstrapped" '{"native_uv_route_prepared":true,"credentials_transferred":false}' >/dev/null
