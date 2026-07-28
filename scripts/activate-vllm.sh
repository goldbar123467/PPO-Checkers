#!/usr/bin/env bash
# shellcheck shell=bash
_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_script_dir/configure-env.sh"
source "$ML_LAB_HOME/.venv-vllm/bin/activate"
unset _script_dir
