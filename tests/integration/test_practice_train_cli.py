"""Practice-only training orchestration without running the powered arena."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

from checkers import training_cli
from checkers.config import RunConfig
from checkers.eval.ballots import BallotSet
from checkers.eval.policy_eval import PracticePolicyEvaluation
from checkers.logging_wandb import RunMetadata
from checkers.metric_history import MetricHistoryWriter
from checkers.train import TrainingSession
from checkers.trainer_state import TrainerState
from checkers.training_cli import (
    PracticeEvaluationResult,
    _print_approval_gate,
    _print_heartbeat,
    _print_periodic_evaluation,
    run_training,
)

FINAL_LOGGING_STEPS = 2
PRACTICE_GAMES = 432


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="practice-cli-unit",
        seed=0,
        stage="practice",
        device="cpu",
        total_updates=1,
        schedule_horizon_updates=6_144,
        duration_seconds=None,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        periodic_every=1,
        checkpoint_every=1,
        eval_games=432,
        periodic_games=432,
        wandb_mode="disabled",
    )


class _Logger:
    def log(self, metrics: object, *, state: TrainerState) -> None:
        del metrics
        state.advance_logging_step()

    def assert_complete(self, required_keys: object = None) -> None:
        del required_keys

    def log_artifact(self, **kwargs: object) -> None:
        del kwargs

    def finish(self, *, exit_code: int) -> None:
        del exit_code


def _metadata() -> RunMetadata:
    return RunMetadata(
        git_sha="a" * 40,
        git_dirty=False,
        hostname="unit",
        python_version="3",
        torch_version="2",
        numpy_version="2",
        gymnasium_version="1",
        wandb_version="1",
        device_name="cpu",
        deterministic=True,
    )


def test_practice_run_uses_final_ballot_eval_local_history_and_compact_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config()
    config_path = tmp_path / "practice.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    scalar_metrics = {
        f"eval/vs_{anchor}{suffix}": 1.0
        for anchor in ("random", "minimax2")
        for suffix in ("", "_ci_low", "_ci_high", "_games")
    }
    evaluation = cast(
        PracticePolicyEvaluation,
        SimpleNamespace(scalar_metrics=scalar_metrics),
    )

    def create_logger(**kwargs: object) -> _Logger:
        state = cast(TrainerState, kwargs["state"])
        state.wandb_run_id = "practice-unit-run"
        return _Logger()

    def evaluate(**kwargs: object) -> PracticeEvaluationResult:
        session = cast(TrainingSession, kwargs["session"])
        logger = cast(_Logger, kwargs["logger"])
        history = cast(MetricHistoryWriter, kwargs["history"])
        output = cast(Path, kwargs["output_directory"])
        training_metrics = cast(dict[str, float], kwargs["training_metrics"])
        kind = cast(str, kwargs["kind"])
        logging_step = session.state.logging_step
        history.append(
            kind=f"{kind}_evaluation",
            metrics=scalar_metrics,
            state=session.state,
            logging_step=logging_step,
        )
        logger.log(scalar_metrics, state=session.state)
        path = output / "evaluations" / f"update-{session.state.update_idx:06d}-{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        _print_periodic_evaluation(
            session=session,
            training_metrics=training_metrics,
            evaluation_metrics=scalar_metrics,
            kind=kind,
        )
        return PracticeEvaluationResult(path=path, evaluation=evaluation, wall_seconds=1.0)

    monkeypatch.setattr(training_cli, "collect_run_metadata", lambda **_kwargs: _metadata())
    monkeypatch.setattr(training_cli, "scan_repository_for_credentials", lambda _path: ())
    monkeypatch.setattr(training_cli, "create_wandb_logger", create_logger)
    monkeypatch.setattr(
        training_cli,
        "load_ballot_set",
        lambda _path: cast(BallotSet, object()),
    )
    monkeypatch.setattr(training_cli, "_run_practice_evaluation", evaluate)

    result = run_training(
        config_path=config_path,
        output_directory=tmp_path / "run",
        resume_path=None,
        max_updates=None,
    )

    assert result.end_update == 1
    assert result.logging_step == FINAL_LOGGING_STEPS
    assert result.evaluation_path is not None
    history = [
        json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in history] == ["training", "final_evaluation"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["evaluation_games"] == PRACTICE_GAMES
    output = capsys.readouterr().out
    assert "final update=1 transitions=1" in output
    assert "checkpoint update=1 transitions=1" in output
    assert "policy_loss=" in output
    assert "vs_minimax2=1" in output


def test_practice_gate_and_heartbeat_lines_are_numeric(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = TrainerState(
        global_step=8_388_608,
        update_idx=1_024,
        elapsed_training_seconds=123.5,
    )
    session = cast(TrainingSession, SimpleNamespace(state=state))
    metrics = {
        "charts/SPS": 42.0,
        "policy/normalized_entropy": 0.04,
        "env/draw_rate": 0.96,
        "train/explained_variance": -0.1,
    }

    _print_heartbeat(session=session, training_metrics=metrics)
    _print_approval_gate(
        session=session,
        training_metrics=metrics,
        minimax_scores=[0.2, 0.3, 0.4],
    )

    output = capsys.readouterr().out
    assert "heartbeat update=1024 transitions=8388608 elapsed=123.5s SPS=42" in output
    assert "approval_gate status=PAUSED update=1024" in output
    assert "eval/vs_minimax2_ma3=0.3" in output
    assert "env/draw_rate=0.96" in output
