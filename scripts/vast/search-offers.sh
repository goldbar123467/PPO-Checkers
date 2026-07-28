#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

profile=""
gpu=""
gpu_count=""
min_vram=""
min_total_vram=""
min_disk=""
min_reliability=""
min_cuda=""
min_down=""
min_up=""
max_hourly=""
min_hours=""
limit=20
raw_output=""
while (($#)); do
  case "$1" in
    --profile) profile="${2:?}"; shift 2 ;;
    --gpu) gpu="${2:?}"; shift 2 ;;
    --gpu-count) gpu_count="${2:?}"; shift 2 ;;
    --min-vram) min_vram="${2:?}"; shift 2 ;;
    --min-total-vram) min_total_vram="${2:?}"; shift 2 ;;
    --min-disk) min_disk="${2:?}"; shift 2 ;;
    --min-reliability) min_reliability="${2:?}"; shift 2 ;;
    --min-cuda) min_cuda="${2:?}"; shift 2 ;;
    --min-download) min_down="${2:?}"; shift 2 ;;
    --min-upload) min_up="${2:?}"; shift 2 ;;
    --max-hourly) max_hourly="${2:?}"; shift 2 ;;
    --min-hours) min_hours="${2:?}"; shift 2 ;;
    --limit) limit="${2:?}"; shift 2 ;;
    --raw-output) raw_output="${2:?}"; shift 2 ;;
    -h|--help)
      printf 'Usage: %s --profile NAME [--gpu MODEL] [--gpu-count N] [--min-vram GB] [--min-total-vram GB] [--min-disk GB] [--min-reliability SCORE] [--min-cuda VERSION] [--min-download MBPS] [--min-upload MBPS] [--max-hourly USD] [--min-hours HOURS] [--limit N] [--raw-output FILE]\n' "$0"
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -n "$profile" ]] || die "--profile is required."
resolve_profile "$profile"
require_cmd vastai
require_vast_auth

# Fail closed if the installed CLI no longer advertises fields used below.
schema="$(vastai search offers --help)"
for field in gpu_name num_gpus gpu_ram gpu_total_ram disk_space reliability cuda_vers inet_down inet_up direct_port_count dph duration; do
  grep -q "$field" <<<"$schema" || die "Installed Vast CLI schema no longer advertises required field: $field"
done

gpu="${gpu:-$(yaml_scalar gpu_models)}"
gpu_count="${gpu_count:-$(yaml_scalar gpu_count)}"
min_vram="${min_vram:-$(yaml_scalar min_vram_per_gpu_gb)}"
min_total_vram="${min_total_vram:-$(yaml_scalar min_total_vram_gb)}"
min_disk="${min_disk:-$(yaml_scalar disk_gb)}"
min_reliability="${min_reliability:-$(yaml_scalar min_reliability)}"
min_cuda="${min_cuda:-$(yaml_scalar min_cuda_version)}"
min_down="${min_down:-$(yaml_scalar min_download_mbps)}"
min_up="${min_up:-$(yaml_scalar min_upload_mbps)}"
[[ "$gpu" =~ ^[A-Za-z0-9_,.+-]+$ ]] || die "GPU model filter contains unsupported characters; use comma-separated underscore names."
for value in "$gpu_count" "$min_vram" "$min_total_vram" "$min_disk" "$min_reliability" "$min_cuda" "$min_down" "$min_up"; do
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "Profile contains a non-numeric offer constraint."
done

IFS=',' read -r -a gpu_items <<<"$gpu"
if ((${#gpu_items[@]} == 1)); then
  gpu_query="gpu_name=${gpu_items[0]}"
else
  gpu_query="gpu_name in [$(IFS=,; printf '%s' "${gpu_items[*]}")]"
fi
query="rentable=true verified=true $gpu_query num_gpus=$gpu_count gpu_ram>=$min_vram gpu_total_ram>=$min_total_vram disk_space>=$min_disk reliability>=$min_reliability cuda_vers>=$min_cuda inet_down>=$min_down inet_up>=$min_up direct_port_count>=1"
if [[ -n "$max_hourly" ]]; then
  require_positive_number "maximum hourly price" "$max_hourly"
  query+=" dph<=$max_hourly"
fi
if [[ -n "$min_hours" ]]; then
  require_positive_number "minimum hours" "$min_hours"
  days="$(python3 -c 'import sys; print(float(sys.argv[1])/24)' "$min_hours")"
  query+=" duration>=$days"
fi
[[ "$limit" =~ ^[0-9]+$ ]] || die "--limit must be an integer."

tmp="$(mktemp "$ML_LAB_HOME/cache/temporary/vast-offers.XXXXXX" 2>/dev/null || mktemp)"
trap 'rm -f -- "$tmp"' EXIT
vastai search offers "$query" --order 'reliability-,dlperf_usd-' --limit "$limit" --raw >"$tmp"
python3 "$VAST_SCRIPT_DIR/_offer.py" display --input "$tmp"
if [[ -n "$raw_output" ]]; then
  [[ ! -e "$raw_output" ]] || die "Refusing to overwrite: $raw_output"
  install -m 600 -- "$tmp" "$raw_output"
  info "Raw non-secret offer data saved to: $raw_output"
fi
