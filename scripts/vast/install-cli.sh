#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

require_cmd uv
uv tool install --upgrade vastai
vastai --version
vastai search offers --help >/dev/null
info "Vast.ai CLI is installed as an isolated uv tool."
