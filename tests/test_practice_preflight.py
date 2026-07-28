"""Pure acceptance accounting for the measured practice preflight."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

from checkers import practice_preflight
from checkers.config import RunConfig
from checkers.eval.ballots import BallotSet, OpeningBallot
from checkers.eval.policy_eval import PracticePolicyEvaluation
from checkers.practice_preflight import (
    ChildRun,
    _run_child,
    loss_equivalence,
    run_preflight,
)
from checkers.train import TrainingSession
from checkers.trainer_state import TrainerState

LOSS_VALUE_COUNT = 2
PREFLIGHT_LOSS_VALUES = 40
FULL_EVALUATION_GAMES = 864
ARENA_EVALUATION_GAMES = 64


def _history(policy: float, value: float) -> dict[int, dict[str, float]]:
    return {
        1: {
            "train/policy_loss": policy,
            "train/value_loss": value,
        }
    }


def test_loss_equivalence_uses_float64_bits_and_reports_numeric_delta() -> None:
    first = _history(0.0, 1.0)
    same = _history(0.0, 1.0)
    different = _history(-0.0, math.nextafter(1.0, 2.0))

    assert loss_equivalence(first, same) == (LOSS_VALUE_COUNT, 0, 0.0)
    compared, mismatches, max_delta = loss_equivalence(first, different)

    assert compared == LOSS_VALUE_COUNT
    assert mismatches == LOSS_VALUE_COUNT
    assert max_delta == math.nextafter(1.0, 2.0) - 1.0


def test_loss_equivalence_rejects_different_update_coverage() -> None:
    with pytest.raises(ValueError, match="update indices"):
        loss_equivalence(_history(0.0, 1.0), {})


def _practice_config() -> RunConfig:
    return RunConfig(
        experiment_id="preflight-unit",
        seed=0,
        stage="practice",
        device="cpu",
        total_updates=6_144,
        schedule_horizon_updates=6_144,
        duration_seconds=None,
        num_envs=64,
        num_steps=128,
        periodic_every=96,
        checkpoint_every=256,
        eval_games=432,
        periodic_games=432,
        wandb_mode="online",
    )


def _metric_history(path: Path, config: RunConfig) -> None:
    records = []
    for update_idx in range(1, 21):
        records.append(
            json.dumps(
                {
                    "kind": "training",
                    "update_idx": update_idx,
                    "metrics": {
                        "train/policy_loss": update_idx / 100.0,
                        "train/value_loss": update_idx / 50.0,
                        "train/lr": config.learning_rate
                        * (1.0 - (update_idx - 1) / config.schedule_horizon_updates),
                        "mask/sample_legality_violations": 0.0,
                        "mask/oracle_disagreements": 0.0,
                        "mask/empty_mask_count": 0.0,
                    },
                },
                sort_keys=True,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_run_child_keeps_key_out_of_arguments_and_preserves_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "child"
    checkpoint = output / "checkpoints" / "update-000020.pt"
    metrics = output / "metrics.jsonl"
    manifest = output / "manifest-000020.json"
    captured: dict[str, object] = {}

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        metrics.write_text("{}\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "metrics_history": str(metrics),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(subprocess, "run", execute)
    result = _run_child(
        repository=tmp_path,
        config_path=tmp_path / "config.yaml",
        output_directory=output,
        mode="online",
        max_updates=20,
        end_update=20,
    )

    assert result.checkpoint == checkpoint
    assert cast(dict[str, str], captured["environment"])["WANDB_MODE"] == "online"
    assert all("API_KEY" not in argument for argument in cast(list[str], captured["command"]))
    assert (output / "preflight-000020.stdout.log").read_text(encoding="utf-8") == "done\n"


def test_run_preflight_reports_all_numeric_acceptance_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _practice_config()
    config_path = tmp_path / "practice.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    ballots = cast(tuple[OpeningBallot, ...], tuple(range(216)))
    ballot_set = BallotSet(
        ballots=ballots,
        sha256="a" * 64,
        source_sequence_count=302,
        source_sequences_sha256="b" * 64,
        distinct_first_moves=7,
        transposition_examples=(),
    )

    def child(**kwargs: object) -> ChildRun:
        output = cast(Path, kwargs["output_directory"])
        end_update = cast(int, kwargs["end_update"])
        mode = cast(str, kwargs["mode"])
        checkpoint = output / "checkpoints" / f"update-{end_update:06d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        metrics = output / "metrics.jsonl"
        _metric_history(metrics, config)
        return ChildRun(
            output_directory=output,
            manifest={
                "wall_seconds": 10.0,
                "wandb_run_id": "remote-run" if mode == "online" else "offline-run",
            },
            checkpoint=checkpoint,
            metrics=metrics,
        )

    def evaluation(**kwargs: object) -> PracticePolicyEvaluation:
        games = len(cast(tuple[object, ...], kwargs["ballots"])) * 2
        match = SimpleNamespace(games=games, records=tuple(range(games)))
        return cast(
            PracticePolicyEvaluation,
            SimpleNamespace(random_match=match, minimax_match=match),
        )

    state = TrainerState(
        global_step=20 * config.batch_size,
        update_idx=20,
    )
    session = cast(
        TrainingSession,
        SimpleNamespace(config=config, state=state, network=object()),
    )
    monkeypatch.setenv("WANDB_API_KEY", "environment-only-unit-key")
    monkeypatch.setattr(practice_preflight, "_run_child", child)
    monkeypatch.setattr(
        TrainingSession,
        "resume",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(practice_preflight, "load_ballot_set", lambda _path: ballot_set)
    monkeypatch.setattr(practice_preflight, "evaluate_practice_policy", evaluation)
    monkeypatch.setattr(
        practice_preflight,
        "_sequential_practice_matches",
        lambda **kwargs: (
            SimpleNamespace(records=tuple(range(len(kwargs["ballot_set"].ballots) * 2))),
            SimpleNamespace(records=tuple(range(len(kwargs["ballot_set"].ballots) * 2))),
        ),
    )

    report = run_preflight(
        repository=tmp_path,
        config_path=config_path,
        output_directory=tmp_path / "preflight",
    )

    assert report["accepted"] == 1
    assert report["acceptance_failure_count"] == 0
    assert report["wandb_loss_values_compared"] == PREFLIGHT_LOSS_VALUES
    assert report["resume_loss_values_compared"] == PREFLIGHT_LOSS_VALUES
    assert report["evaluation_games"] == FULL_EVALUATION_GAMES
    assert report["arena_benchmark_games"] == ARENA_EVALUATION_GAMES
    assert report["arena_record_mismatches"] == 0
    assert report["lr_absolute_error"] == 0.0
