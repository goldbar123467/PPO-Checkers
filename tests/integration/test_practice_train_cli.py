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
    PRACTICE_APPROVAL_GATE_UPDATE,
    PracticeEvaluationResult,
    _print_approval_gate,
    _print_heartbeat,
    _print_periodic_evaluation,
    run_training,
)

FINAL_LOGGING_STEPS = 2
PRACTICE_GAMES = 432
PRACTICE_TOTAL_ANCHOR_GAMES = 4
PRACTICE_SEQUENCE_COUNT = 302


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


def test_practice_evaluation_writes_timing_scalars_and_replay_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    state = TrainerState(global_step=1, update_idx=1, elapsed_training_seconds=2.0)
    alerts: list[dict[str, float]] = []

    def record_alerts(metrics: dict[str, float]) -> None:
        alerts.append(metrics)

    session = cast(
        TrainingSession,
        SimpleNamespace(
            config=config,
            state=state,
            network=object(),
            check_evaluation_alerts=record_alerts,
        ),
    )
    match = SimpleNamespace(games=2, records=("game-1", "game-2"))
    scalar_metrics = {
        "eval/vs_random": 0.75,
        "eval/vs_random_ci_low": 0.5,
        "eval/vs_random_ci_high": 1.0,
        "eval/vs_random_games": 2.0,
        "eval/vs_minimax2": 0.5,
        "eval/vs_minimax2_ci_low": 0.25,
        "eval/vs_minimax2_ci_high": 0.75,
        "eval/vs_minimax2_games": 2.0,
    }
    evaluation = cast(
        PracticePolicyEvaluation,
        SimpleNamespace(
            scalar_metrics=scalar_metrics,
            game_rows=({"game": 1},),
            random_match=match,
            minimax_match=match,
        ),
    )
    ballot_set = BallotSet(
        ballots=(),
        sha256="a" * 64,
        source_sequence_count=302,
        source_sequences_sha256="b" * 64,
        distinct_first_moves=7,
        transposition_examples=(),
    )
    logger = _Logger()
    history = MetricHistoryWriter(path=tmp_path / "metrics.jsonl", next_logging_step=0)
    monkeypatch.setattr(
        training_cli,
        "evaluate_practice_policy",
        lambda **_kwargs: evaluation,
    )

    result = training_cli._run_practice_evaluation(
        session=session,
        logger=logger,  # type: ignore[arg-type]
        history=history,
        output_directory=tmp_path,
        ballot_set=ballot_set,
        training_metrics={
            "charts/SPS": 10.0,
            "train/policy_loss": 0.1,
            "train/approx_kl": 0.01,
            "train/clipfrac": 0.02,
            "train/explained_variance": 0.3,
            "policy/normalized_entropy": 0.9,
            "env/draw_rate": 0.4,
        },
        kind="periodic",
    )

    assert result.path.is_file()
    record = json.loads(result.path.read_text(encoding="utf-8"))
    assert record["schema"] == "CHECKERS_PRACTICE_EVALUATION_1"
    assert record["total_games"] == PRACTICE_TOTAL_ANCHOR_GAMES
    assert record["source_sequence_count"] == PRACTICE_SEQUENCE_COUNT
    history_record = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert history_record["metrics"]["eval/wall_seconds"] >= 0.0
    assert history_record["metrics"]["eval/games_per_second"] >= 0.0
    assert alerts == [scalar_metrics]


def test_fresh_long_invocation_stops_at_the_update_1024_approval_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = RunConfig(
        **{
            **asdict(_config()),
            "total_updates": 6_144,
            "periodic_every": 96,
            "checkpoint_every": 256,
        }
    )
    config_path = tmp_path / "practice.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    state = TrainerState(
        global_step=(PRACTICE_APPROVAL_GATE_UPDATE - 1) * config.batch_size,
        update_idx=PRACTICE_APPROVAL_GATE_UPDATE - 1,
        elapsed_training_seconds=10.0,
    )
    metrics = {
        "charts/SPS": 100.0,
        "train/policy_loss": 0.1,
        "train/value_loss": 0.2,
        "train/approx_kl": 0.01,
        "train/clipfrac": 0.02,
        "train/explained_variance": 0.3,
        "policy/normalized_entropy": 0.4,
        "env/draw_rate": 0.5,
    }

    class GateSession:
        def __init__(self) -> None:
            self.config = config
            self.state = state

        def run_update(self) -> SimpleNamespace:
            self.state.advance_update(config, elapsed_seconds=1.0)
            return SimpleNamespace(metrics=metrics)

        def save_checkpoint(self, path: Path, **_kwargs: object) -> SimpleNamespace:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"gate-checkpoint")
            path.with_suffix(".pt.sha256").write_text("c" * 64 + "\n", encoding="ascii")
            return SimpleNamespace(sha256="c" * 64)

    gate_session = cast(TrainingSession, GateSession())
    evaluation = cast(
        PracticePolicyEvaluation,
        SimpleNamespace(scalar_metrics={"eval/vs_minimax2": 0.25}),
    )

    def create_logger(**kwargs: object) -> _Logger:
        cast(TrainerState, kwargs["state"]).wandb_run_id = "gate-run"
        return _Logger()

    def gate_evaluation(**kwargs: object) -> PracticeEvaluationResult:
        session = cast(TrainingSession, kwargs["session"])
        logger = cast(_Logger, kwargs["logger"])
        history = cast(MetricHistoryWriter, kwargs["history"])
        output = cast(Path, kwargs["output_directory"])
        logging_step = session.state.logging_step
        scalar = {"eval/vs_minimax2": 0.25}
        history.append(
            kind="approval_gate_evaluation",
            metrics=scalar,
            state=session.state,
            logging_step=logging_step,
        )
        logger.log(scalar, state=session.state)
        path = output / "evaluations" / "update-001024-approval_gate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return PracticeEvaluationResult(path=path, evaluation=evaluation, wall_seconds=1.0)

    monkeypatch.setattr(training_cli, "collect_run_metadata", lambda **_kwargs: _metadata())
    monkeypatch.setattr(training_cli, "scan_repository_for_credentials", lambda _path: ())
    monkeypatch.setattr(TrainingSession, "create", lambda **_kwargs: gate_session)
    monkeypatch.setattr(training_cli, "create_wandb_logger", create_logger)
    monkeypatch.setattr(
        training_cli,
        "load_ballot_set",
        lambda _path: cast(BallotSet, object()),
    )
    monkeypatch.setattr(training_cli, "_run_practice_evaluation", gate_evaluation)

    result = run_training(
        config_path=config_path,
        output_directory=tmp_path / "run",
        resume_path=None,
        max_updates=None,
    )

    assert result.end_update == PRACTICE_APPROVAL_GATE_UPDATE
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "paused_for_approval"
    assert manifest["approval_gate_update"] == PRACTICE_APPROVAL_GATE_UPDATE
    output = capsys.readouterr().out
    assert "checkpoint update=1024" in output
    assert "heartbeat update=1024" in output
    assert "approval_gate status=PAUSED update=1024" in output
    assert "eval/vs_minimax2_ma3=0.25" in output
