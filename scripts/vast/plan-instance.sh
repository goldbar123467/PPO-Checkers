#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

profile=""
offer_id=""
offer_json=""
max_hourly=""
max_hours=""
output=""
while (($#)); do
  case "$1" in
    --profile) profile="${2:?}"; shift 2 ;;
    --offer) offer_id="${2:?}"; shift 2 ;;
    --offer-json) offer_json="${2:?}"; shift 2 ;;
    --max-hourly) max_hourly="${2:?}"; shift 2 ;;
    --max-hours) max_hours="${2:?}"; shift 2 ;;
    --output) output="${2:?}"; shift 2 ;;
    -h|--help)
      printf 'Usage: %s --profile NAME (--offer ID | --offer-json FILE) --max-hourly USD --max-hours HOURS [--output FILE]\n' "$0"
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -n "$profile" && -n "$max_hourly" && -n "$max_hours" ]] || die "--profile, --max-hourly, and --max-hours are required."
[[ -n "$offer_id" || -n "$offer_json" ]] || die "Specify exactly one of --offer or --offer-json."
[[ -z "$offer_id" || -z "$offer_json" ]] || die "Specify only one of --offer or --offer-json."
require_positive_number "maximum hourly price" "$max_hourly"
require_positive_number "maximum runtime" "$max_hours"
resolve_profile "$profile"

tmp=""
if [[ -n "$offer_id" ]]; then
  require_instance_id "$offer_id"
  require_cmd vastai
  require_vast_auth
  tmp="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-plan.XXXXXX" 2>/dev/null || mktemp)"
  trap 'rm -f -- "$tmp"' EXIT
  vastai search offers "id=$offer_id rentable=true" --limit 1 --raw >"$tmp"
  offer_json="$tmp"
fi
[[ -f "$offer_json" ]] || die "Offer JSON not found: $offer_json"
if [[ -z "$output" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output="$VAST_MANIFEST_DIR/plans/${stamp}-$(basename "$PROFILE_PATH" .yaml)-offer-${offer_id:-fixture}.json"
fi
python3 "$VAST_SCRIPT_DIR/_offer.py" plan --input "$offer_json" --profile "$PROFILE_PATH" \
  ${offer_id:+--offer-id "$offer_id"} --max-hourly "$max_hourly" --max-hours "$max_hours" --output "$output"
info "No resource was created. Review the immutable plan before running create-instance.sh."
