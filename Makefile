SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

LAB := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TRAIN_PY := $(LAB)/.venv-train/bin/python
TRAIN_BIN := $(LAB)/.venv-train/bin
VLLM_BIN := $(LAB)/.venv-vllm/bin
CHECKERS_LINT_PATHS := src/checkers tests/test_phase0_scaffold.py tests/rules tests/env \
	tests/rl tests/eval tests/property tests/metamorphic tests/integration tests/golden
MODEL ?=
PORT ?= 8000
export PYTHONPATH := $(LAB)/src
export WANDB_MODE ?= offline

.PHONY: help format format-check lint types test mutate fuzz-ci fuzz perft check smoke train eval \
	doctor ruff mypy smoke-train disk jupyter tensorboard hf-login kaggle-test serve-vllm \
	test-vllm ollama-start ollama-stop ollama-test clean-dry-run

help:
	@echo "Checkers: format lint types test mutate fuzz-ci fuzz perft check smoke train eval"
	@echo "ML Lab:  doctor smoke-train disk jupyter tensorboard hf-login kaggle-test"
	@echo "         serve-vllm test-vllm ollama-start ollama-stop ollama-test clean-dry-run"

format:
	@cd $(LAB) && $(TRAIN_BIN)/ruff format $(CHECKERS_LINT_PATHS)

format-check:
	@cd $(LAB) && $(TRAIN_BIN)/ruff format --check $(CHECKERS_LINT_PATHS)

lint:
	@cd $(LAB) && $(TRAIN_BIN)/ruff check $(CHECKERS_LINT_PATHS)

types:
	@cd $(LAB) && $(TRAIN_BIN)/mypy --strict $(CHECKERS_LINT_PATHS)

doctor:
	@$(LAB)/scripts/doctor.sh

test:
	@cd $(LAB) && $(TRAIN_PY) -m pytest

mutate:
	@cd $(LAB) && $(TRAIN_BIN)/mutmut run --paths-to-mutate=src/checkers/rules

fuzz-ci:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q tests/property tests/test_phase0_scaffold.py

fuzz:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q tests/property tests/metamorphic

perft:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q tests/rules -k 'perft or differential'

check: format-check lint types test fuzz-ci

smoke:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/train.py --config configs/checkers-smoke.yaml

train:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/train.py

eval:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/evaluate.py

ruff: lint

mypy:
	@cd $(LAB) && $(TRAIN_BIN)/mypy src/ml_lab

smoke-train:
	@$(LAB)/scripts/smoke-train.sh

disk:
	@$(LAB)/scripts/disk-report.sh

jupyter:
	@source $(LAB)/scripts/configure-env.sh && exec $(TRAIN_BIN)/jupyter-lab \
		--ip=127.0.0.1 --port=8888 --no-browser --notebook-dir=$(LAB)

tensorboard:
	@source $(LAB)/scripts/configure-env.sh && exec $(TRAIN_BIN)/tensorboard \
		--host=127.0.0.1 --port=6006 --logdir=$(LAB)/runs/tensorboard

hf-login:
	@$(LAB)/scripts/hf-login.sh

kaggle-test:
	@$(LAB)/scripts/kaggle-test.sh

serve-vllm:
	@test -n "$(MODEL)" || { echo "Usage: make serve-vllm MODEL=<model-id> [PORT=8000]" >&2; exit 2; }
	@$(LAB)/scripts/serve-vllm.sh "$(MODEL)" "$(PORT)"

test-vllm:
	@$(LAB)/scripts/test-vllm-api.sh "$(PORT)"

ollama-start:
	@$(LAB)/scripts/start-ollama.sh

ollama-stop:
	@$(LAB)/scripts/stop-ollama.sh

ollama-test:
	@$(LAB)/scripts/test-ollama.sh

clean-dry-run:
	@$(LAB)/scripts/cleanup-cache.sh
