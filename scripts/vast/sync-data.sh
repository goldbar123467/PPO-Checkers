#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

instance=""; dataset=""; remote_name=""; execute=false
while (($#)); do
  case "$1" in
    --instance) instance="${2:?}"; shift 2 ;;
    --dataset) dataset="${2:?}"; shift 2 ;;
    --remote-name) remote_name="${2:?}"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) printf 'Usage: %s --instance ID --dataset PATH [--remote-name NAME] [--execute]\n' "$0"; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
require_instance_id "$instance"
[[ -e "$dataset" ]] || die "Dataset path does not exist: $dataset"
dataset="$(realpath -- "$dataset")"
case "$dataset" in
  "$ML_LAB_HOME/.secrets"/*|"$HOME/.ssh"/*|"$VAST_CONFIG_HOME"/*|"$ML_LAB_HOME/.env") die "Refusing to transfer a credential-bearing path." ;;
esac
remote_name="${remote_name:-$(basename -- "$dataset")}"; [[ "$remote_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe remote dataset name."
local_fp="$(python3 "$VAST_SCRIPT_DIR/_fingerprint.py" "$dataset")"
python3 - "$local_fp" <<'PY'
import json,sys
v=json.loads(sys.argv[1]); print(f"Local dataset: {v['file_count']} files, {v['total_bytes']} bytes, records={v['record_count']}, sha256={v['sha256']}")
PY
if [[ "$execute" != true ]]; then info "DRY RUN: no data transferred. Add --execute after review."; exit 0; fi
ssh_connection "$instance"; transport="$(rsync_ssh_command)"; remote_dir="/workspace/data/$remote_name"
remote_shell "$instance" "set -Eeuo pipefail; test ! -e $(printf %q "$remote_dir") || { echo 'Remote dataset target already exists; refusing overwrite.' >&2; exit 1; }; mkdir -p $(printf %q "$remote_dir")"
rsync -az -e "$transport" -- "$dataset" "$SSH_TARGET:$remote_dir/"
remote_fp="$(remote_shell "$instance" "python3 /workspace/ml-lab/scripts/vast/_fingerprint.py $(printf %q "$remote_dir")" | tail -n1)"
python3 - "$local_fp" "$remote_fp" <<'PY'
import json,sys
a,b=map(json.loads,sys.argv[1:])
for key in ('sha256','file_count','total_bytes','record_count'):
    if a[key] != b[key]: raise SystemExit(f"dataset {key} mismatch: local={a[key]} remote={b[key]}")
print("Dataset hash, file count, byte count, and applicable record count verified.")
PY
manifest="$(manifest_for_instance "$instance")"; run_id="$(manifest_run_id "$manifest")"
payload="$(python3 - "$local_fp" "$remote_dir" <<'PY'
import json,sys
v=json.loads(sys.argv[1]); print(json.dumps({'data_sync_status':'verified','dataset_sha256':v['sha256'],'dataset_file_count':v['file_count'],'dataset_total_bytes':v['total_bytes'],'dataset_record_count':v['record_count'],'remote_dataset_path':sys.argv[2]},separators=(',',':')))
PY
)"
manifest_append "$run_id" "data_sync_verified" "$payload" >/dev/null
