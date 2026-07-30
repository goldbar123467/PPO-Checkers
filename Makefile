SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV_PROJECT_ENVIRONMENT ?= .venv
VENV_BIN := $(abspath $(UV_PROJECT_ENVIRONMENT))/bin
PYTHON := $(VENV_BIN)/python
CHECKERS_LINT_PATHS := src/checkers \
	tests/test_checkpoint.py tests/test_logging_wandb.py tests/test_metric_history.py \
	tests/test_recovery.py tests/test_monitor.py tests/test_practice_preflight.py \
	tests/test_run_runtime.py tests/test_system_metrics.py \
	tests/rules tests/env tests/agents tests/web tests/rl tests/eval tests/property \
	tests/metamorphic tests/integration tests/golden \
	scripts/build_published_transcripts.py scripts/differential_rules.py \
	scripts/run_rule_mutation_challenges.py scripts/fuzz_environment.py \
	scripts/evaluate_baselines.py scripts/generate_ballots.py \
	scripts/generate_dev_tactics.py scripts/preflight_practice.py scripts/train.py \
	scripts/build_checkers_release_report.py scripts/export_checkers_policy.py \
	scripts/serve_checkers_web.py scripts/recover_checkers_run.py \
	scripts/audit_recovery_smoke.py scripts/monitor_run.py

export PYTHONPATH := $(PROJECT_ROOT)/src
export WANDB_MODE ?= offline

.PHONY: help format format-check lint types test mutate fuzz-ci fuzz fuzz-env perft \
	check smoke train eval

help:
	@echo "PPO Checkers: format lint types test mutate fuzz-ci fuzz fuzz-env perft check smoke train eval"

format:
	@cd $(PROJECT_ROOT) && $(VENV_BIN)/ruff format $(CHECKERS_LINT_PATHS)

format-check:
	@cd $(PROJECT_ROOT) && $(VENV_BIN)/ruff format --check $(CHECKERS_LINT_PATHS)

lint:
	@cd $(PROJECT_ROOT) && $(VENV_BIN)/ruff check $(CHECKERS_LINT_PATHS)

types:
	@cd $(PROJECT_ROOT) && $(VENV_BIN)/mypy --strict $(CHECKERS_LINT_PATHS)

test:
	@cd $(PROJECT_ROOT) && $(PYTHON) -m pytest

mutate:
	@cd $(PROJECT_ROOT) && $(VENV_BIN)/mutmut run

fuzz-ci:
	@cd $(PROJECT_ROOT) && $(PYTHON) -m pytest -q --no-cov tests/property

fuzz:
	@cd $(PROJECT_ROOT) && $(PYTHON) -m pytest -q --no-cov tests/property tests/metamorphic

fuzz-env:
	@cd $(PROJECT_ROOT) && $(PYTHON) scripts/fuzz_environment.py --steps 50000 \
		--output runs/validation/environment-fuzz-50k.json

perft:
	@cd $(PROJECT_ROOT) && $(PYTHON) -m pytest -q --no-cov tests/rules -k 'perft or differential'

check: format-check lint types test fuzz-ci

smoke:
	@cd $(PROJECT_ROOT) && WANDB_MODE=offline $(PYTHON) scripts/train.py \
		--config configs/checkers-smoke.yaml

train:
	@cd $(PROJECT_ROOT) && WANDB_MODE=offline $(PYTHON) scripts/train.py

eval:
	@cd $(PROJECT_ROOT) && WANDB_MODE=offline $(PYTHON) scripts/evaluate_baselines.py
