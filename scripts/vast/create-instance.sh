#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

plan=""
execute=false
accept_cost=false
while (($#)); do
  case "$1" in
    --plan) plan="${2:?}"; shift 2 ;;
    --execute) execute=true; shift ;;
    --accept-cost) accept_cost=true; shift ;;
    -h|--help)
      printf 'Usage: %s --plan PLAN.json [--execute --accept-cost]\n' "$0"
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -f "$plan" ]] || die "--plan must name an existing plan."
plan_json="$(python3 "$VAST_SCRIPT_DIR/_offer.py" verify-plan --plan "$plan")"
python3 - "$plan" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
print(f"Offer: {p['offer_id']} | {p['gpu_model']} x{p['gpu_count']} | ${p['hourly_cost_usd']:.4f}/hr")
print(f"Approved cap: ${p['approved_maximum_hourly_usd']:.4f}/hr for {p['approved_maximum_runtime_hours']} hr")
print(f"Maximum theoretical cost: ${p['maximum_theoretical_cost_usd']:.4f}")
print(f"Image: {p['docker_image']} | disk: {p['disk_allocation_gb']} GB | profile: {p['profile']}")
PY
if ! confirm_execute "$execute" "$accept_cost"; then
  exit 0
fi
require_cmd vastai
require_vast_auth

mapfile -t fields < <(python3 - "$plan" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
for k in ('offer_id','docker_image','disk_allocation_gb','profile','hourly_cost_usd','approved_maximum_hourly_usd','approved_maximum_runtime_hours','maximum_theoretical_cost_usd','gpu_model','gpu_count','vram_per_gpu_gb','machine_id','profile_sha256','training_mode','intended_model_size','accelerate_config'):
    print(p.get(k,''))
PY
)
offer_id="${fields[0]}"; image="${fields[1]}"; disk="${fields[2]}"; profile="${fields[3]}"
[[ "$image" != *REPLACE_WITH* ]] || die "Plan image is still a placeholder. Build and publish the trusted lab image or choose the pinned NVIDIA debugging image."

# Re-query at execution time and ensure the offer still exists. The plan validator already
# enforces the approved cap; create never substitutes another offer.
live="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-live-offer.XXXXXX" 2>/dev/null || mktemp)"
response="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-create.XXXXXX" 2>/dev/null || mktemp)"
trap 'rm -f -- "$live" "$response"' EXIT
vastai search offers "id=$offer_id rentable=true" --limit 1 --raw >"$live"
python3 "$VAST_SCRIPT_DIR/_offer.py" display --input "$live" >/dev/null || die "Offer is no longer rentable; no replacement will be launched."

run_id="vast-$(date -u +%Y%m%dT%H%M%SZ)-offer-$offer_id"
payload="$(python3 - "$plan" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
keys=('offer_id','machine_id','gpu_model','gpu_count','vram_per_gpu_gb','hourly_cost_usd','approved_maximum_hourly_usd','approved_maximum_runtime_hours','maximum_theoretical_cost_usd','docker_image','profile','profile_sha256','training_mode','intended_model_size','accelerate_config','remote_paths')
print(json.dumps({k:p.get(k) for k in keys}, separators=(',',':')))
PY
)"
manifest_append "$run_id" "creation_approved" "$payload" >/dev/null
vastai create instance "$offer_id" --image "$image" --disk "$disk" --ssh --direct \
  --label "ml-lab-$run_id" --cancel-unavail --raw >"$response"
instance_id="$(python3 - "$response" <<'PY'
import json, sys
v=json.load(open(sys.argv[1], encoding='utf-8'))
i=v.get('new_contract') or v.get('id') or (v.get('instance') or {}).get('id')
if not i: raise SystemExit('create response did not contain an instance ID')
print(int(i))
PY
)"
created_payload="$(python3 - "$instance_id" <<'PY'
import json, sys
print(json.dumps({'remote_instance_id': int(sys.argv[1]), 'instance_creation_time_recorded': True}, separators=(',',':')))
PY
)"
manifest_append "$run_id" "instance_created" "$created_payload" >/dev/null
info "Instance created: $instance_id"
info "Local run ID: $run_id"
info "No replacement or runtime extension will occur automatically."
