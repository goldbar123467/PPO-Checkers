SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

LAB := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TRAIN_PY := $(LAB)/.venv-train/bin/python
TRAIN_BIN := $(LAB)/.venv-train/bin
VLLM_BIN := $(LAB)/.venv-vllm/bin
CHECKERS_LINT_PATHS := src/checkers tests/test_phase0_scaffold.py tests/test_checkpoint.py \
	tests/test_logging_wandb.py tests/test_metric_history.py tests/test_recovery.py \
	tests/test_monitor.py tests/test_practice_preflight.py tests/test_run_runtime.py \
	tests/test_system_metrics.py \
	tests/rules tests/env tests/agents \
	tests/rl tests/eval tests/property tests/metamorphic tests/integration tests/golden \
	scripts/build_published_transcripts.py scripts/differential_rules.py \
	scripts/run_rule_mutation_challenges.py scripts/fuzz_environment.py \
	scripts/evaluate_baselines.py scripts/generate_ballots.py scripts/preflight_practice.py \
	scripts/train.py \
	scripts/recover_checkers_run.py scripts/audit_recovery_smoke.py scripts/monitor_run.py
MODEL ?=
PORT ?= 8000
export PYTHONPATH := $(LAB)/src
export WANDB_MODE ?= offline

.PHONY: help format format-check lint types test mutate fuzz-ci fuzz fuzz-env fuzz-env-gate perft \
	check smoke train eval \
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
	@cd $(LAB) && $(TRAIN_BIN)/mutmut run

fuzz-ci:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q --no-cov tests/property tests/test_phase0_scaffold.py

fuzz:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q --no-cov tests/property tests/metamorphic

fuzz-env:
	@cd $(LAB) && $(TRAIN_PY) scripts/fuzz_environment.py --steps 50000 \
		--output reports/phase4_environment_fuzz_50k_seed20260728.json

fuzz-env-gate:
	@cd $(LAB) && $(TRAIN_PY) scripts/fuzz_environment.py --steps 5000000 \
		--output reports/phase4_environment_fuzz_5m_seed20260728.json

perft:
	@cd $(LAB) && $(TRAIN_PY) -m pytest -q --no-cov tests/rules -k 'perft or differential'

check: format-check lint types test fuzz-ci

smoke:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/train.py --config configs/checkers-smoke.yaml

train:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/train.py

eval:
	@cd $(LAB) && WANDB_MODE=offline $(TRAIN_PY) scripts/evaluate_baselines.py

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
