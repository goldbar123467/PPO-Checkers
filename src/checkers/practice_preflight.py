"""Measured 20-update acceptance preflight for the seed-0 PPO practice run."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from checkers.agents.minimax_agent import MinimaxAgent
from checkers.agents.policy_agent import PolicyAgent
from checkers.agents.random_agent import RandomAgent
from checkers.config import load_run_config
from checkers.eval.arena import AgentSpec, MatchResult, play_ballot_match
from checkers.eval.ballots import BallotSet, load_ballot_set
from checkers.eval.baseline_run import atomic_write_bytes
from checkers.eval.policy_eval import evaluate_practice_policy
from checkers.rl.determinism import derive_stream_seed
from checkers.schedules import current_lr
from checkers.train import TrainingSession

PREFLIGHT_UPDATES = 20
SPLIT_UPDATES = 10
PRACTICE_UPDATES = 6_144
PERIODIC_EVALUATIONS = 64
ARENA_BENCHMARK_BALLOTS = 16
LOSS_KEYS = ("train/policy_loss", "train/value_loss")
MASK_KEYS = (
    "mask/sample_legality_violations",
    "mask/oracle_disagreements",
    "mask/empty_mask_count",
)


@dataclass(frozen=True, slots=True)
class ChildRun:
    """Paths and parsed terminal manifest for one isolated training process."""

    output_directory: Path
    manifest: dict[str, object]
    checkpoint: Path
    metrics: Path


def _atomic_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _run_child(  # noqa: PLR0913
    *,
    repository: Path,
    config_path: Path,
    output_directory: Path,
    mode: str,
    max_updates: int,
    end_update: int,
    resume: Path | None = None,
) -> ChildRun:
    command = [
        sys.executable,
        str(repository / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_directory),
        "--max-updates",
        str(max_updates),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    environment = os.environ.copy()
    environment["WANDB_MODE"] = mode
    environment.setdefault("WANDB_INIT_TIMEOUT", "30")
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_directory / f"preflight-{end_update:06d}.stdout.log", completed.stdout)
    _atomic_text(output_directory / f"preflight-{end_update:06d}.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} child run failed with exit code {completed.returncode}; "
            f"see {output_directory}"
        )
    manifest_path = output_directory / f"manifest-{end_update:06d}.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError("training manifest must be a mapping")
    manifest = cast(dict[str, object], manifest_value)
    checkpoint = Path(cast(str, manifest["checkpoint"]))
    metrics = Path(cast(str, manifest["metrics_history"]))
    if not checkpoint.is_file() or not metrics.is_file():
        raise RuntimeError("child run omitted its checkpoint or local metric history")
    return ChildRun(
        output_directory=output_directory,
        manifest=manifest,
        checkpoint=checkpoint,
        metrics=metrics,
    )


def _training_metrics(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["kind"] != "training":
            continue
        update_idx = value["update_idx"]
        metrics = value["metrics"]
        if isinstance(update_idx, bool) or not isinstance(update_idx, int):
            raise ValueError("training history update_idx must be an integer")
        if not isinstance(metrics, dict):
            raise ValueError("training history metrics must be a mapping")
        result[update_idx] = {
            str(key): float(metric) for key, metric in metrics.items()
        }
    return result


def loss_equivalence(
    first: Mapping[int, Mapping[str, float]],
    second: Mapping[int, Mapping[str, float]],
) -> tuple[int, int, float]:
    """Return compared values, bitwise mismatches, and maximum absolute loss delta."""

    if set(first) != set(second):
        raise ValueError("loss histories cover different update indices")
    compared = 0
    mismatches = 0
    max_absolute_delta = 0.0
    for update_idx in sorted(first):
        for key in LOSS_KEYS:
            first_value = float(first[update_idx][key])
            second_value = float(second[update_idx][key])
            if not math.isfinite(first_value) or not math.isfinite(second_value):
                raise ValueError("loss histories must contain finite values")
            compared += 1
            mismatches += int(
                struct.pack("!d", first_value) != struct.pack("!d", second_value)
            )
            max_absolute_delta = max(max_absolute_delta, abs(first_value - second_value))
    return compared, mismatches, max_absolute_delta


def _position_seeded_minimax(*, max_plies: int) -> AgentSpec:
    return AgentSpec(
        name="minimax(2)",
        factory=lambda seed: MinimaxAgent(depth=2, seed=seed, max_plies=max_plies),
        position_seeded=True,
    )


def _sequential_practice_matches(
    *,
    session: TrainingSession,
    ballot_set: BallotSet,
    seed: int,
) -> tuple[MatchResult, MatchResult]:
    current = AgentSpec(
        name="current",
        factory=lambda value: PolicyAgent(
            network=session.network,
            mode="greedy",
            seed=value,
            name="current",
        ),
    )
    random_match = play_ballot_match(
        first=current,
        second=AgentSpec(name="random", factory=lambda value: RandomAgent(seed=value)),
        ballots=ballot_set.ballots,
        seed=derive_stream_seed(seed, 0),
        max_plies=session.config.max_plies,
        repetition_draws=session.config.repetition_draws,
    )
    minimax_match = play_ballot_match(
        first=current,
        second=_position_seeded_minimax(max_plies=session.config.max_plies),
        ballots=ballot_set.ballots,
        seed=derive_stream_seed(seed, 1),
        max_plies=session.config.max_plies,
        repetition_draws=session.config.repetition_draws,
    )
    return random_match, minimax_match


def _failure_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def run_preflight(  # noqa: PLR0914, PLR0915
    *,
    repository: Path,
    config_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Execute all practice acceptance checks and return their numeric evidence."""

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be present in the environment for online preflight")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("preflight output directory must be absent or empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    config = load_run_config(config_path.read_text(encoding="utf-8"))
    if config.stage != "practice" or config.seed != 0:
        raise ValueError("preflight requires the seed-0 practice configuration")

    offline = _run_child(
        repository=repository,
        config_path=config_path,
        output_directory=output_directory / "offline-20",
        mode="offline",
        max_updates=PREFLIGHT_UPDATES,
        end_update=PREFLIGHT_UPDATES,
    )
    online = _run_child(
        repository=repository,
        config_path=config_path,
        output_directory=output_directory / "online-20",
        mode="online",
        max_updates=PREFLIGHT_UPDATES,
        end_update=PREFLIGHT_UPDATES,
    )
    split_directory = output_directory / "resume-20"
    split_first = _run_child(
        repository=repository,
        config_path=config_path,
        output_directory=split_directory,
        mode="offline",
        max_updates=SPLIT_UPDATES,
        end_update=SPLIT_UPDATES,
    )
    split_final = _run_child(
        repository=repository,
        config_path=config_path,
        output_directory=split_directory,
        mode="offline",
        max_updates=SPLIT_UPDATES,
        end_update=PREFLIGHT_UPDATES,
        resume=split_first.checkpoint,
    )

    offline_metrics = _training_metrics(offline.metrics)
    online_metrics = _training_metrics(online.metrics)
    resumed_metrics = _training_metrics(split_final.metrics)
    wandb_compared, wandb_mismatches, wandb_max_delta = loss_equivalence(
        offline_metrics,
        online_metrics,
    )
    resume_compared, resume_mismatches, resume_max_delta = loss_equivalence(
        offline_metrics,
        resumed_metrics,
    )

    final_metrics = offline_metrics[PREFLIGHT_UPDATES]
    session = TrainingSession.resume(config=config, checkpoint_path=offline.checkpoint)
    ballot_set = load_ballot_set(repository / "data" / "ballots_v1.json")
    evaluation_seed = derive_stream_seed(config.seed, 2_000_000 + PREFLIGHT_UPDATES)
    eval_started = time.perf_counter()
    full_evaluation = evaluate_practice_policy(
        network=session.network,
        ballots=ballot_set.ballots,
        seed=evaluation_seed,
        max_plies=config.max_plies,
        repetition_draws=config.repetition_draws,
    )
    eval_wall_seconds = time.perf_counter() - eval_started
    eval_games = full_evaluation.random_match.games + full_evaluation.minimax_match.games

    benchmark_set = BallotSet(
        ballots=ballot_set.ballots[:ARENA_BENCHMARK_BALLOTS],
        sha256=ballot_set.sha256,
        source_sequence_count=ballot_set.source_sequence_count,
        source_sequences_sha256=ballot_set.source_sequences_sha256,
        distinct_first_moves=ballot_set.distinct_first_moves,
        transposition_examples=ballot_set.transposition_examples,
    )
    benchmark_seed = derive_stream_seed(config.seed, 3_000_000)
    batched_started = time.perf_counter()
    batched = evaluate_practice_policy(
        network=session.network,
        ballots=benchmark_set.ballots,
        seed=benchmark_seed,
        max_plies=config.max_plies,
        repetition_draws=config.repetition_draws,
    )
    batched_seconds = time.perf_counter() - batched_started
    unbatched_started = time.perf_counter()
    unbatched_random, unbatched_minimax = _sequential_practice_matches(
        session=session,
        ballot_set=benchmark_set,
        seed=benchmark_seed,
    )
    unbatched_seconds = time.perf_counter() - unbatched_started
    arena_games = ARENA_BENCHMARK_BALLOTS * 4
    arena_record_mismatches = sum(
        first != second
        for first, second in zip(
            (*batched.random_match.records, *batched.minimax_match.records),
            (*unbatched_random.records, *unbatched_minimax.records),
            strict=True,
        )
    )

    training_wall_20 = float(cast(float, offline.manifest["wall_seconds"]))
    extrapolated_training_wall = training_wall_20 * PRACTICE_UPDATES / PREFLIGHT_UPDATES
    extrapolated_eval_wall = eval_wall_seconds * PERIODIC_EVALUATIONS
    eval_percent = 100.0 * extrapolated_eval_wall / extrapolated_training_wall
    boundary_lr = current_lr(config, session.state)
    expected_lr = config.learning_rate * (1.0 - PREFLIGHT_UPDATES / PRACTICE_UPDATES)
    online_failures = _failure_count(online.output_directory / "wandb_failures.jsonl")
    online_run_id = str(online.manifest["wandb_run_id"])

    report: dict[str, object] = {
        "schema": "CHECKERS_PRACTICE_PREFLIGHT_1",
        "updates": PREFLIGHT_UPDATES,
        "seed": config.seed,
        "schedule_horizon_updates": config.schedule_horizon_updates,
        "wandb_loss_values_compared": wandb_compared,
        "wandb_loss_bitwise_mismatches": wandb_mismatches,
        "wandb_loss_max_absolute_delta": wandb_max_delta,
        "wandb_offline_online_bitwise_equal": int(wandb_mismatches == 0),
        "wandb_online_failure_count": online_failures,
        "wandb_online_remote_run": int(not online_run_id.startswith("local-")),
        "wandb_online_run_id": online_run_id,
        "resume_loss_values_compared": resume_compared,
        "resume_loss_bitwise_mismatches": resume_mismatches,
        "resume_loss_max_absolute_delta": resume_max_delta,
        "checkpoint_resume_loss_equivalent": int(resume_mismatches == 0),
        "illegal_action_count": final_metrics[MASK_KEYS[0]],
        "oracle_disagreement_count": final_metrics[MASK_KEYS[1]],
        "empty_mask_count": final_metrics[MASK_KEYS[2]],
        "lr_logged_update_20": final_metrics["train/lr"],
        "lr_boundary_update_20": boundary_lr,
        "lr_expected_6144_horizon": expected_lr,
        "lr_absolute_error": abs(boundary_lr - expected_lr),
        "training_wall_20_seconds": training_wall_20,
        "training_wall_6144_extrapolated_seconds": extrapolated_training_wall,
        "evaluation_wall_seconds": eval_wall_seconds,
        "evaluation_games": eval_games,
        "evaluation_games_per_second": eval_games / eval_wall_seconds,
        "periodic_evaluations": PERIODIC_EVALUATIONS,
        "evaluation_wall_6144_extrapolated_seconds": extrapolated_eval_wall,
        "evaluation_wall_percent_of_training_wall": eval_percent,
        "arena_benchmark_ballots": ARENA_BENCHMARK_BALLOTS,
        "arena_benchmark_games": arena_games,
        "arena_batched_wall_seconds": batched_seconds,
        "arena_unbatched_wall_seconds": unbatched_seconds,
        "arena_batched_games_per_second": arena_games / batched_seconds,
        "arena_unbatched_games_per_second": arena_games / unbatched_seconds,
        "arena_record_mismatches": arena_record_mismatches,
        "offline_checkpoint": str(offline.checkpoint),
        "online_checkpoint": str(online.checkpoint),
        "resume_checkpoint": str(split_final.checkpoint),
    }
    failures = (
        wandb_mismatches
        + resume_mismatches
        + online_failures
        + int(online_run_id.startswith("local-"))
        + int(any(final_metrics[key] != 0.0 for key in MASK_KEYS))
        + int(boundary_lr != expected_lr)
        + arena_record_mismatches
    )
    report["acceptance_failure_count"] = failures
    report["accepted"] = int(failures == 0)
    _atomic_text(
        output_directory / "preflight_report.json",
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    if failures:
        raise RuntimeError(
            f"practice preflight produced {failures} acceptance failures; "
            f"see {output_directory / 'preflight_report.json'}"
        )
    return report
