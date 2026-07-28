"""Offline W&B initialization, monotonic logging, and completeness contracts."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from checkers.config import RunConfig
from checkers.logging_wandb import (
    GAME_TABLE_COLUMNS,
    PAYOFF_TABLE_COLUMNS,
    PROJECT_NAME,
    RunMetadata,
    WandbLogger,
    collect_run_metadata,
    create_wandb_logger,
    game_table,
    payoff_matrix_table,
    scan_repository_for_credentials,
)
from checkers.metrics import REQUIRED_METRIC_KEYS
from checkers.trainer_state import TrainerState

RESUMED_LOGGING_STEP = 9
ARTIFACT_FILE_COUNT = 2


class FakeRun:
    def __init__(self, run_id: str = "offline-id") -> None:
        self.id = run_id
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, object], int, bool]] = []
        self.finished = False
        self.artifacts: list[Any] = []

    def log(self, data: dict[str, object], *, step: int, commit: bool) -> None:
        self.logged.append((data, step, commit))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = exit_code == 0

    def log_artifact(self, artifact: object) -> None:
        self.artifacts.append(artifact)


class FailingRun(FakeRun):
    def log(self, data: dict[str, object], *, step: int, commit: bool) -> None:
        del data, step, commit
        random.random()
        np.random.random()
        torch.rand(1)
        raise ConnectionError("simulated network failure")

    def finish(self, *, exit_code: int = 0) -> None:
        del exit_code
        raise ConnectionError("simulated finish failure")

    def log_artifact(self, artifact: object) -> None:
        del artifact
        raise ConnectionError("simulated artifact failure")


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="wandb-unit",
        seed=23,
        device="cpu",
        total_updates=1,
        schedule_horizon_updates=1,
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


def test_environment_can_force_online_config_through_unchanged_offline_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return FakeRun()

    create_wandb_logger(
        config=RunConfig(**{**asdict(_config()), "wandb_mode": "online"}),
        state=TrainerState(),
        metadata=_metadata(),
        stamp="override",
        directory=tmp_path,
        run_factory=factory,
    )

    assert captured["mode"] == "offline"
    assert "api_key" not in captured


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


def test_w1_recovery_summary_is_namespaced_and_does_not_change_run_config() -> None:
    config = _config()
    state = TrainerState(wandb_run_id="source-id")
    captured: dict[str, Any] = {}
    fake = FakeRun(run_id="source-id")

    def factory(**kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return fake

    create_wandb_logger(
        config=config,
        state=state,
        metadata=_metadata(),
        stamp="recovery",
        run_factory=factory,
        additional_summary={
            "recovery/is_recovery": True,
            "recovery/checkpoint_sha256": "a" * 64,
        },
    )

    assert captured["config"] == asdict(config)
    assert captured["id"] == "source-id"
    assert fake.summary["recovery/is_recovery"] is True
    assert fake.summary["recovery/checkpoint_sha256"] == "a" * 64


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


def test_w1_real_wandb_offline_run_persists_complete_local_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "true")
    config = _config()
    state = TrainerState()

    logger = create_wandb_logger(
        config=config,
        state=state,
        metadata=_metadata(),
        stamp="20260728Toffline",
        directory=tmp_path,
    )
    logger.log({key: 0.0 for key in REQUIRED_METRIC_KEYS}, state=state)
    logger.assert_complete()
    logger.finish(exit_code=0)

    run_directories = tuple((tmp_path / "wandb").glob("offline-run-*"))
    assert len(run_directories) == 1
    assert tuple(run_directories[0].glob("run-*.wandb"))
    assert state.logging_step == 1
    assert state.wandb_run_id


def test_w1_payoff_and_rendered_game_tables_preserve_literal_rows() -> None:
    payoff_rows = (
        {
            "row_agent": "current",
            "column_agent": "initial",
            "wins": 1,
            "draws": 1,
            "losses": 0,
            "games": 2,
            "score": 0.75,
        },
    )
    game_rows = (
        {
            "match": "current_vs_initial",
            "game_index": 0,
            "perspective_agent": "current",
            "perspective_result": "draw",
            "red_agent": "current",
            "white_agent": "initial",
            "winner": "DRAW",
            "termination_reason": "repetition",
            "steps": 12,
            "completed_moves": 10,
            "moves": "9-13 22-18",
            "actions": "32 88",
            "red_seed": 1,
            "white_seed": 2,
            "environment_seed": 3,
        },
    )

    payoff = payoff_matrix_table(payoff_rows)
    games = game_table(game_rows)

    assert payoff.columns == list(PAYOFF_TABLE_COLUMNS)
    assert payoff.data == [[payoff_rows[0][column] for column in PAYOFF_TABLE_COLUMNS]]
    assert games.columns == list(GAME_TABLE_COLUMNS)
    assert games.data == [[game_rows[0][column] for column in GAME_TABLE_COLUMNS]]


def test_w1_versioned_artifact_contains_declared_local_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update-000010.pt"
    digest = tmp_path / "update-000010.pt.sha256"
    checkpoint.write_bytes(b"weights")
    digest.write_text("0" * 64 + "\n", encoding="ascii")
    fake = FakeRun()
    logger = WandbLogger(run=fake, initial_logging_step=0)

    logger.log_artifact(
        name="checkpoint-unit-update-000010",
        artifact_type="model",
        files=(checkpoint, digest),
        metadata={"update_idx": 10},
    )

    assert len(fake.artifacts) == 1
    artifact = fake.artifacts[0]
    assert artifact.name == "checkpoint-unit-update-000010"
    assert artifact.type == "model"
    assert len(artifact.manifest.entries) == ARTIFACT_FILE_COUNT


def test_wandb_init_failure_records_locally_and_returns_a_noop_logger(tmp_path: Path) -> None:
    state = TrainerState()

    def fail_init(**_kwargs: object) -> object:
        raise ConnectionError("simulated init failure")

    logger = create_wandb_logger(
        config=_config(),
        state=state,
        metadata=_metadata(),
        stamp="failure",
        directory=tmp_path,
        run_factory=fail_init,
    )
    logger.log({"train/policy_loss": 1.0}, state=state)
    logger.finish(exit_code=0)

    records = [
        json.loads(line)
        for line in (tmp_path / "wandb_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["operation"] for record in records] == ["init"]
    assert records[0]["error_type"] == "ConnectionError"
    assert state.wandb_run_id.startswith("local-")
    assert state.logging_step == 1


def test_wandb_failures_never_propagate_and_logging_restores_all_global_rngs(
    tmp_path: Path,
) -> None:
    random.seed(11)
    np.random.seed(13)
    torch.manual_seed(17)
    python_before = random.getstate()
    numpy_before = cast(tuple[Any, ...], np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    state = TrainerState()
    logger = WandbLogger(
        run=FailingRun(),
        initial_logging_step=0,
        failure_path=tmp_path / "wandb_failures.jsonl",
    )
    artifact_file = tmp_path / "checkpoint.pt"
    artifact_file.write_bytes(b"checkpoint")

    logger.log({"train/policy_loss": 1.0}, state=state)
    logger.log_artifact(
        name="failed-artifact",
        artifact_type="model",
        files=(artifact_file,),
        metadata={"update_idx": 1},
    )
    logger.finish(exit_code=0)

    assert random.getstate() == python_before
    numpy_after = cast(tuple[Any, ...], np.random.get_state())
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert state.logging_step == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "wandb_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["operation"] for record in records] == ["log", "artifact", "finish"]


def test_wandb_logger_rejects_invalid_local_contract_inputs(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="run"):
        WandbLogger(run=object(), initial_logging_step=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="initial_logging_step"):
        WandbLogger(run=None, initial_logging_step=True)
    with pytest.raises(ValueError, match="non-negative"):
        WandbLogger(run=None, initial_logging_step=-1)
    with pytest.raises(TypeError, match="failure_path"):
        WandbLogger(run=None, initial_logging_step=0, failure_path="bad")  # type: ignore[arg-type]

    logger = WandbLogger(run=None, initial_logging_step=0)
    state = TrainerState()
    with pytest.raises(TypeError, match="mapping"):
        logger.log([], state=state)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TrainerState"):
        logger.log({"metric": 1.0}, state=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        logger.log({}, state=state)
    with pytest.raises(TypeError, match="metric names"):
        logger.log(cast(dict[str, object], {1: 1.0}), state=state)
    logger.log({"metric": 1.0}, state=state)
    state.logging_step = 0
    with pytest.raises(ValueError, match="monotonic"):
        logger.log({"metric": 2.0}, state=state)

    with pytest.raises(TypeError, match="summary values"):
        logger.update_summary([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="summary keys"):
        logger.update_summary(cast(dict[str, object], {1: 1.0}))
    logger.update_summary({"valid": 1.0})
    with pytest.raises(TypeError, match="required_keys"):
        logger.assert_complete(cast(frozenset[str], {"bad"}))
    with pytest.raises(TypeError, match="required_keys"):
        logger.assert_complete(cast(frozenset[str], frozenset({""})))

    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"x")
    with pytest.raises(ValueError, match="name"):
        logger.log_artifact(name="", artifact_type="model", files=(artifact,), metadata={})
    with pytest.raises(ValueError, match="artifact_type"):
        logger.log_artifact(name="name", artifact_type="", files=(artifact,), metadata={})
    with pytest.raises(ValueError, match="non-empty tuple"):
        logger.log_artifact(name="name", artifact_type="model", files=(), metadata={})
    with pytest.raises(TypeError, match="metadata"):
        logger.log_artifact(
            name="name",
            artifact_type="model",
            files=(artifact,),
            metadata=[],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Paths"):
        logger.log_artifact(
            name="name",
            artifact_type="model",
            files=cast(tuple[Path, ...], ("bad",)),
            metadata={},
        )
    with pytest.raises(ValueError, match="does not exist"):
        logger.log_artifact(
            name="name",
            artifact_type="model",
            files=(tmp_path / "missing",),
            metadata={},
        )
    logger.log_artifact(name="name", artifact_type="model", files=(artifact,), metadata={})
    with pytest.raises(TypeError, match="exit_code"):
        logger.finish(exit_code=True)
    logger.finish(exit_code=0)


def test_wandb_summary_failure_and_missing_failure_path_fall_back_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingSummary(dict[str, object]):
        def update(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ConnectionError("summary failure")

    run = FakeRun()
    run.summary = FailingSummary()
    logger = WandbLogger(run=run, initial_logging_step=0)

    logger.update_summary({"metric": 1.0})

    assert "wandb_failure operation=summary error_type=ConnectionError" in capsys.readouterr().err


def test_create_wandb_logger_validates_boundaries_and_malformed_init_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = {
        "config": _config(),
        "state": TrainerState(),
        "metadata": _metadata(),
        "stamp": "valid",
        "directory": tmp_path,
        "run_factory": lambda **_kwargs: FakeRun(),
    }
    invalid_cases: tuple[tuple[dict[str, object], type[Exception], str], ...] = (
        ({"config": object()}, TypeError, "config"),
        ({"state": object()}, TypeError, "state"),
        ({"metadata": object()}, TypeError, "metadata"),
        ({"stamp": ""}, ValueError, "stamp"),
        ({"directory": "bad"}, TypeError, "directory"),
        ({"additional_summary": []}, TypeError, "additional_summary"),
        (
            {"additional_summary": cast(dict[str, object], {1: 1.0})},
            TypeError,
            "summary keys",
        ),
    )
    for overrides, error_type, message in invalid_cases:
        with pytest.raises(error_type, match=message):
            create_wandb_logger(**{**valid, **overrides})  # type: ignore[arg-type]

    monkeypatch.setenv("WANDB_MODE", "invalid")
    with pytest.raises(ValueError, match="WANDB_MODE"):
        create_wandb_logger(**valid)  # type: ignore[arg-type]
    monkeypatch.delenv("WANDB_MODE")

    for factory in (
        lambda **_kwargs: None,
        lambda **_kwargs: FakeRun(run_id=""),
    ):
        state = TrainerState()
        logger = create_wandb_logger(
            config=_config(),
            state=state,
            metadata=_metadata(),
            stamp="malformed",
            directory=tmp_path,
            run_factory=factory,
        )
        assert state.wandb_run_id.startswith("local-")
        logger.finish(exit_code=0)


def test_tables_metadata_and_scanner_reject_invalid_boundaries(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="tuple"):
        payoff_matrix_table([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        payoff_matrix_table(())
    with pytest.raises(TypeError, match="mappings"):
        payoff_matrix_table(cast(tuple[dict[str, object], ...], (object(),)))
    with pytest.raises(ValueError, match="fields"):
        payoff_matrix_table(({"wrong": 1},))
    with pytest.raises(TypeError, match="config"):
        collect_run_metadata(config=object(), repository=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="repository"):
        collect_run_metadata(config=_config(), repository="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Path"):
        scan_repository_for_credentials("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="directory"):
        scan_repository_for_credentials(tmp_path / "missing")
