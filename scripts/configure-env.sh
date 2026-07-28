#!/usr/bin/env bash
# shellcheck shell=bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead of executing it: source scripts/configure-env.sh" >&2
  exit 2
fi

export ML_LAB_HOME="${ML_LAB_HOME:-$HOME/ml-lab}"
export HF_HOME="$ML_LAB_HOME/cache/huggingface"
export HF_HUB_CACHE="$ML_LAB_HOME/cache/huggingface/hub"
export HF_DATASETS_CACHE="$ML_LAB_HOME/cache/datasets"
export TORCH_HOME="$ML_LAB_HOME/cache/torch"
export TRITON_CACHE_DIR="$ML_LAB_HOME/cache/triton"
export TMPDIR="$ML_LAB_HOME/cache/temporary"
export KAGGLE_CONFIG_DIR="$ML_LAB_HOME/.secrets/kaggle"
export OLLAMA_MODELS="$ML_LAB_HOME/cache/ollama"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$TMPDIR" "$KAGGLE_CONFIG_DIR" "$OLLAMA_MODELS"
