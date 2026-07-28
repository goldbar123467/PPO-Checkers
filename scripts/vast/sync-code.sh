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
[[ -f "$ML_LAB_HOME/uv.lock" && -d "$ML_LAB_HOME/src/ml_lab" ]] || die "Local source or lockfile is missing."
local_fp="$(python3 "$VAST_SCRIPT_DIR/_code_fingerprint.py")"
info "Local code fingerprint: $(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["sha256"])' "$local_fp")"
if [[ "$execute" != true ]]; then
  info "DRY RUN: would transfer only source, lockfiles, configs, and Accelerate files; no .git, environments, data, models, caches, runs, or secrets."
  exit 0
fi
require_cmd rsync
ssh_connection "$instance"
rsync_transport="$(rsync_ssh_command)"
rsync -az --prune-empty-dirs -e "$rsync_transport" \
  --include='/pyproject.toml' --include='/uv.lock' --include='/README.md' \
  --include='/src/***' --include='/configs/***' \
  --include='/scripts/' --include='/scripts/configure-env.sh' --include='/scripts/vast/' \
  --include='/scripts/vast/_fingerprint.py' --include='/scripts/vast/_code_fingerprint.py' \
  --include='/cloud/' --include='/cloud/vast/' --include='/cloud/vast/accelerate/***' \
  --exclude='*' -- "$ML_LAB_HOME/" "$SSH_TARGET:/workspace/ml-lab/"
remote_shell "$instance" 'set -Eeuo pipefail; cd /workspace/ml-lab; export PATH=/root/.local/bin:$PATH; uv sync --frozen --no-dev --python 3.12; mkdir -p /workspace/cache/temporary; python scripts/vast/_code_fingerprint.py' >"$VAST_LOG_DIR/instance-$instance-code-sync.log"
remote_fp="$(tail -n1 "$VAST_LOG_DIR/instance-$instance-code-sync.log")"
python3 - "$local_fp" "$remote_fp" <<'PY'
import json, sys
local, remote = map(json.loads, sys.argv[1:])
if local != remote:
    raise SystemExit(f"code fingerprint mismatch: local={local} remote={remote}")
print("Code fingerprint verified.")
PY
manifest="$(manifest_for_instance "$instance")"; run_id="$(manifest_run_id "$manifest")"
git_commit="$(git -C "$ML_LAB_HOME" rev-parse HEAD 2>/dev/null || true)"
git_dirty="$(git -C "$ML_LAB_HOME" status --porcelain 2>/dev/null | wc -l)"
payload="$(python3 - "$local_fp" "$git_commit" "$git_dirty" <<'PY'
import json, sys
fp=json.loads(sys.argv[1]); print(json.dumps({'code_sync_status':'verified','source_sha256':fp['sha256'],'source_file_count':fp['file_count'],'source_total_bytes':fp['total_bytes'],'git_commit':sys.argv[2] or None,'git_dirty_path_count':int(sys.argv[3])}, separators=(',',':')))
PY
)"
manifest_append "$run_id" "code_sync_verified" "$payload" >/dev/null
