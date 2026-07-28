"""Offline W&B initialization, monotonic logging, and completeness contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from checkers.config import RunConfig
from checkers.logging_wandb import (
    PROJECT_NAME,
    RunMetadata,
    WandbLogger,
    create_wandb_logger,
    scan_repository_for_credentials,
)
from checkers.metrics import REQUIRED_METRIC_KEYS
from checkers.trainer_state import TrainerState

RESUMED_LOGGING_STEP = 9


class FakeRun:
    def __init__(self, run_id: str = "offline-id") -> None:
        self.id = run_id
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, object], int, bool]] = []
        self.finished = False

    def log(self, data: dict[str, object], *, step: int, commit: bool) -> None:
        self.logged.append((data, step, commit))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = exit_code == 0


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="wandb-unit",
        seed=23,
        device="cpu",
        total_timesteps=2,
        duration_seconds=None,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        eval_games=2,
    )


def _metadata() -> RunMetadata:
    return RunMetadata(
        git_sha="0123456789abcdef",
        git_dirty=True,
        hostname="test-host",
        python_version="3.test",
        torch_version="2.test",
        numpy_version="2.test",
        gymnasium_version="1.test",
        wandb_version="0.test",
        device_name="cpu",
        deterministic=True,
    )


def test_w1_initialization_uses_exact_name_tags_config_and_provenance() -> None:
    config = _config()
    state = TrainerState()
    captured: dict[str, Any] = {}
    fake = FakeRun()

    def factory(**kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return fake

    logger = create_wandb_logger(
        config=config,
        state=state,
        metadata=_metadata(),
        stamp="20260728T041500-0400",
        run_factory=factory,
    )

    assert isinstance(logger, WandbLogger)
    assert captured == {
        "project": PROJECT_NAME,
        "name": "phase-7-0123456-seed23-20260728T041500-0400",
        "tags": ("phase-7", "seed-23", "arm-A0", "stage-A"),
        "config": asdict(config),
        "mode": "offline",
    }
    assert fake.summary == {
        "provenance/git_sha": "0123456789abcdef",
        "provenance/git_dirty": True,
        "provenance/hostname": "test-host",
        "provenance/python_version": "3.test",
        "provenance/torch_version": "2.test",
        "provenance/numpy_version": "2.test",
        "provenance/gymnasium_version": "1.test",
        "provenance/wandb_version": "0.test",
        "provenance/device_name": "cpu",
        "provenance/deterministic": True,
    }
    assert state.wandb_run_id == "offline-id"


def test_w1_resume_uses_saved_id_and_monotonic_explicit_logging_steps() -> None:
    config = _config()
    state = TrainerState(wandb_run_id="saved-id", logging_step=7)
    captured: dict[str, Any] = {}
    fake = FakeRun(run_id="saved-id")

    def factory(**kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return fake

    logger = create_wandb_logger(
        config=config,
        state=state,
        metadata=_metadata(),
        stamp="ignored",
        run_factory=factory,
    )
    logger.log({"train/policy_loss": 1.0}, state=state)
    logger.log({"train/value_loss": 2.0}, state=state)

    assert captured["id"] == "saved-id"
    assert captured["resume"] == "allow"
    assert [record[1] for record in fake.logged] == [7, 8]
    assert all(record[2] for record in fake.logged)
    assert state.logging_step == RESUMED_LOGGING_STEP
    with pytest.raises(ValueError, match="logging step"):
        state.logging_step = 8
        logger.log({"train/entropy": 0.1}, state=state)


def test_w1_completeness_audit_fails_closed_then_accepts_exact_inventory() -> None:
    state = TrainerState()
    fake = FakeRun()
    logger = WandbLogger(run=fake, initial_logging_step=0)

    logger.log({"train/policy_loss": 0.0}, state=state)
    with pytest.raises(RuntimeError, match="missing required metrics"):
        logger.assert_complete()
    logger.log({key: 0.0 for key in REQUIRED_METRIC_KEYS}, state=state)
    logger.assert_complete()
    logger.finish(exit_code=0)

    assert logger.observed_metric_keys == REQUIRED_METRIC_KEYS
    assert fake.finished


def test_w2_repository_scan_rejects_key_shapes_and_credential_files(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text("WANDB_MODE = 'offline'\n", encoding="utf-8")
    assert scan_repository_for_credentials(tmp_path) == ()
    (tmp_path / ".env").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("api_key='A" + "1" * 39 + "'\n", encoding="utf-8")

    findings = scan_repository_for_credentials(tmp_path)

    assert {finding.path.name for finding in findings} == {".env", "bad.txt"}
