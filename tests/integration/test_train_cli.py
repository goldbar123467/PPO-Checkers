"""Tiny offline config-to-W&B-to-checkpoint CLI integration and rerun test."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from checkers.checkpoint import load_checkpoint
from checkers.config import RunConfig
from checkers.rl.networks import CheckersNetwork
from checkers.train import TrainingSession
from checkers.training_cli import run_training

FINAL_UPDATE = 2
TINY_BEST_RESPONSE_GAMES = 2
FIRST_INVOCATION_LOG_STEPS = 2
FINAL_LOG_STEPS = 5


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="offline-cli-unit",
        seed=71,
        device="cpu",
        total_updates=2,
        schedule_horizon_updates=2,
        duration_seconds=None,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        checkpoint_every=1,
        periodic_every=1,
        eval_games=2,
        exploitability_train_games=TINY_BEST_RESPONSE_GAMES,
    )


def test_e1_tiny_offline_cli_run_checkpoints_reloads_and_resumes(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ignore_evaluation_alerts(*_args: object, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(
        TrainingSession,
        "check_evaluation_alerts",
        ignore_evaluation_alerts,
    )
    config = _config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    first = run_training(
        config_path=config_path,
        output_directory=output,
        resume_path=None,
        max_updates=1,
    )

    assert first.start_update == 0
    assert first.end_update == 1
    assert first.checkpoint_path.is_file()
    assert first.checkpoint_path.with_suffix(".pt.sha256").is_file()
    assert first.wandb_run_id
    assert first.logging_step == FIRST_INVOCATION_LOG_STEPS
    assert first.evaluation_path is not None
    first_evaluation = json.loads(first.evaluation_path.read_text(encoding="utf-8"))
    assert first_evaluation["kind"] == "periodic"
    assert first_evaluation["exploitability_status"] == "NOT_EVALUATED"

    second = run_training(
        config_path=config_path,
        output_directory=output,
        resume_path=first.checkpoint_path,
        max_updates=None,
    )

    assert second.start_update == 1
    assert second.end_update == FINAL_UPDATE
    assert second.wandb_run_id == first.wandb_run_id
    assert second.logging_step == FINAL_LOG_STEPS
    assert second.evaluation_path is not None
    assert second.evaluation_path.is_file()
    history = [
        json.loads(line) for line in second.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == second.logging_step
    assert history[-1]["kind"] == "final_evaluation"
    assert history[-1]["metrics"]["eval/exploitability_proxy"] >= 0.0
    assert len(tuple((output / "wandb").glob("offline-run-*"))) == FINAL_UPDATE
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["start_update"] == 1
    assert manifest["end_update"] == FINAL_UPDATE
    assert manifest["evaluation"] == str(second.evaluation_path)
    assert manifest["metrics_history"] == str(second.metrics_path)
    assert manifest["wandb_artifact"].startswith("checkpoint-offline-cli-unit-update-")
    assert manifest["evaluation_games"] == config.eval_games
    evaluation = json.loads(second.evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["kind"] == "final"
    assert evaluation["games_per_match"] == config.eval_games
    assert evaluation["exploitability_status"] == "MEASURED"
    assert evaluation["best_response"]["training_games"] == TINY_BEST_RESPONSE_GAMES
    assert evaluation["best_response"]["training_decisions"] > 0
    assert set(evaluation["metrics"]) >= {"eval/vs_random", "eval/vs_greedy"}

    network = CheckersNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    loaded = load_checkpoint(
        path=second.checkpoint_path,
        expected_config=config,
        network=network,
        optimizer=optimizer,
    )
    assert loaded.state.update_idx == FINAL_UPDATE
    assert loaded.state.logging_step == second.logging_step
    assert loaded.state.wandb_run_id == first.wandb_run_id
