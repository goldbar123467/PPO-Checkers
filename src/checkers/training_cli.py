"""Offline command-line orchestration for reproducible Phase 7 training runs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import yaml

from checkers.config import RunConfig, load_run_config
from checkers.eval.baseline_run import atomic_write_bytes
from checkers.eval.best_response import BestResponseResult, train_short_best_response
from checkers.eval.policy_eval import (
    ExploitabilityEvidence,
    PolicyEvaluation,
    evaluate_development_policy,
    game_rows_from_matches,
)
from checkers.logging_wandb import (
    WandbLogger,
    collect_run_metadata,
    create_wandb_logger,
    game_table,
    payoff_matrix_table,
    scan_repository_for_credentials,
)
from checkers.metric_history import MetricHistoryWriter
from checkers.recovery import validate_recovery_resume_context
from checkers.rl.determinism import derive_stream_seed
from checkers.run_runtime import (
    attach_runtime_run_id,
    finish_runtime_state,
    new_runtime_state,
    write_runtime_state,
)
from checkers.system_metrics import SystemTelemetrySampler
from checkers.train import TrainingSession


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Paths and counters needed to audit or resume one CLI invocation."""

    start_update: int
    end_update: int
    checkpoint_path: Path
    manifest_path: Path
    evaluation_path: Path | None
    metrics_path: Path
    wandb_run_id: str
    logging_step: int


def _max_updates(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_updates must be an integer or None")
    if value < 1:
        raise ValueError("max_updates must be positive")
    return value


def _write_resolved_config(path: Path, config: RunConfig) -> None:
    payload = yaml.safe_dump(asdict(config), sort_keys=False).encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("resolved config already exists with different content")
    atomic_write_bytes(path, payload)


def _manifest_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _source_diff(repository: Path, output_directory: Path) -> Path | None:
    result = subprocess.run(
        ("git", "-C", str(repository), "diff", "--binary", "HEAD"),
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        return None
    path = output_directory / "source.diff"
    atomic_write_bytes(path, result.stdout)
    return path


def _evaluation_record(  # noqa: PLR0913
    *,
    evaluation: PolicyEvaluation,
    kind: str,
    update_idx: int,
    seed: int,
    games: int,
    best_response: BestResponseResult | None,
) -> dict[str, object]:
    return {
        "schema": "CHECKERS_POLICY_EVALUATION_1",
        "kind": kind,
        "update_idx": update_idx,
        "seed": seed,
        "games_per_match": games,
        "exploitability_status": evaluation.exploitability_status,
        "metrics": evaluation.scalar_metrics,
        "payoff_rows": evaluation.payoff_rows,
        "game_rows": evaluation.game_rows,
        "best_response": (
            None
            if best_response is None
            else {
                "training_games": best_response.training_games,
                "training_decisions": best_response.training_decisions,
                "optimizer_steps": best_response.optimizer_steps,
                "score": best_response.evidence.score,
                "score_ci_low": best_response.match.score.low,
                "score_ci_high": best_response.match.score.high,
                "evaluation_games": best_response.match.games,
                "frozen_sha256_before": best_response.frozen_sha256_before,
                "frozen_sha256_after": best_response.frozen_sha256_after,
                "best_response_sha256": best_response.best_response_sha256,
            }
        ),
    }


def _run_evaluation(  # noqa: PLR0913
    *,
    session: TrainingSession,
    logger: WandbLogger,
    history: MetricHistoryWriter,
    output_directory: Path,
    games: int,
    kind: str,
) -> Path:
    stream_offset = 1_000_000 + session.state.update_idx * 2 + (kind == "final")
    evaluation_seed = derive_stream_seed(session.config.seed, stream_offset)
    best_response = (
        train_short_best_response(
            frozen_network=session.network,
            training_games=session.config.exploitability_train_games,
            evaluation_games=games,
            seed=derive_stream_seed(evaluation_seed, 1),
            learning_rate=session.config.learning_rate,
            max_grad_norm=session.config.max_grad_norm,
            max_plies=session.config.max_plies,
            repetition_draws=session.config.repetition_draws,
        )
        if kind == "final"
        else None
    )
    evaluation = evaluate_development_policy(
        network=session.network,
        initial_model_state=session.league.initial.clone_model_state(),
        games=games,
        seed=evaluation_seed,
        max_plies=session.config.max_plies,
        repetition_draws=session.config.repetition_draws,
        exploitability=(
            ExploitabilityEvidence.not_evaluated()
            if best_response is None
            else best_response.evidence
        ),
    )
    if best_response is not None:
        evaluation = replace(
            evaluation,
            game_rows=evaluation.game_rows
            + game_rows_from_matches((("best_response_vs_frozen", best_response.match),)),
        )
    payload: dict[str, object] = dict(evaluation.scalar_metrics)
    payload["eval/payoff_matrix"] = payoff_matrix_table(evaluation.payoff_rows)
    payload["eval/rendered_games"] = game_table(evaluation.game_rows)
    logging_step = session.state.logging_step
    logger.log(payload, state=session.state)
    history.append(
        kind=f"{kind}_evaluation",
        metrics=evaluation.scalar_metrics,
        state=session.state,
        logging_step=logging_step,
    )
    path = output_directory / "evaluations" / f"update-{session.state.update_idx:06d}-{kind}.json"
    atomic_write_bytes(
        path,
        _manifest_bytes(
            _evaluation_record(
                evaluation=evaluation,
                kind=kind,
                update_idx=session.state.update_idx,
                seed=evaluation_seed,
                games=games,
                best_response=best_response,
            )
        ),
    )
    session.check_evaluation_alerts(evaluation.scalar_metrics)
    return path


def run_training(  # noqa: PLR0912, PLR0915
    *,
    config_path: Path,
    output_directory: Path,
    resume_path: Path | None,
    max_updates: int | None,
) -> TrainingRunResult:
    """Run or resume a bounded offline training invocation and persist its final boundary."""

    if not isinstance(config_path, Path):
        raise TypeError("config_path must be a Path")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a Path")
    if resume_path is not None and not isinstance(resume_path, Path):
        raise TypeError("resume_path must be a Path or None")
    checked_max_updates = _max_updates(max_updates)
    config = load_run_config(config_path.read_text(encoding="utf-8"))
    repository = Path(__file__).resolve().parents[2]
    metadata = collect_run_metadata(config=config, repository=repository)
    credential_findings = scan_repository_for_credentials(repository)
    if credential_findings:
        reasons = sorted({finding.reason for finding in credential_findings})
        raise RuntimeError(f"repository credential scan failed: {reasons}")
    output_directory.mkdir(parents=True, exist_ok=True)
    recovery_context = validate_recovery_resume_context(
        output_directory=output_directory,
        resume_path=resume_path,
        current_commit=metadata.git_sha,
        working_tree_clean=not metadata.git_dirty,
    )
    _write_resolved_config(output_directory / "config.resolved.yaml", config)
    session = (
        TrainingSession.create(config=config)
        if resume_path is None
        else TrainingSession.resume(config=config, checkpoint_path=resume_path)
    )
    start_update = session.state.update_idx
    if recovery_context is not None:
        if start_update < recovery_context.checkpoint_update_idx:
            raise RuntimeError("resume checkpoint predates the prepared recovery boundary")
        if session.state.wandb_run_id != recovery_context.source_wandb_run_id:
            raise RuntimeError("resume checkpoint W&B identity disagrees with recovery provenance")
    update_limit = config.total_updates
    if checked_max_updates is not None:
        update_limit = min(update_limit, start_update + checked_max_updates)
    metrics_path = output_directory / "metrics.jsonl"
    history = MetricHistoryWriter(
        path=metrics_path,
        next_logging_step=session.state.logging_step,
    )
    runtime_path = output_directory / "runtime.json"
    runtime_state = new_runtime_state(
        start_update=start_update,
        experiment_id=config.experiment_id,
        seed=config.seed,
        git_sha=metadata.git_sha,
        run_id=session.state.wandb_run_id or None,
        resume_from=resume_path,
    )
    write_runtime_state(runtime_path, runtime_state)
    telemetry = SystemTelemetrySampler(process_pid=runtime_state.pid)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    logger: WandbLogger | None = None
    checkpoints = output_directory / "checkpoints"
    final_checkpoint = resume_path
    evaluation_path: Path | None = None
    artifact_name: str | None = None
    wall_started = time.perf_counter()
    status = "failed"
    runtime_error: str | None = None
    try:
        logger = create_wandb_logger(
            config=config,
            state=session.state,
            metadata=metadata,
            stamp=stamp,
            directory=output_directory,
            additional_summary=(
                None if recovery_context is None else recovery_context.wandb_summary()
            ),
        )
        runtime_state = attach_runtime_run_id(
            runtime_state,
            run_id=session.state.wandb_run_id,
        )
        write_runtime_state(runtime_path, runtime_state)
        while session.state.update_idx < update_limit:
            if (
                config.duration_seconds is not None
                and session.state.elapsed_training_seconds >= config.duration_seconds
            ):
                break
            update = session.run_update()
            logged_metrics = dict(update.metrics)
            logged_metrics.update(telemetry.sample().scalar_metrics())
            logging_step = session.state.logging_step
            logger.log(logged_metrics, state=session.state)
            history.append(
                kind="training",
                metrics=logged_metrics,
                state=session.state,
                logging_step=logging_step,
            )
            if session.state.update_idx % config.eval_every == 0:
                evaluation_path = _run_evaluation(
                    session=session,
                    logger=logger,
                    history=history,
                    output_directory=output_directory,
                    games=config.periodic_eval_games,
                    kind="periodic",
                )
            if (
                session.state.update_idx % config.checkpoint_every == 0
                or session.state.update_idx == update_limit
            ):
                final_checkpoint = checkpoints / f"update-{session.state.update_idx:06d}.pt"
                session.save_checkpoint(
                    final_checkpoint,
                    git_sha=metadata.git_sha,
                    git_dirty=metadata.git_dirty,
                )
        if final_checkpoint is None:
            raise RuntimeError("training invocation produced no checkpoint")
        if not final_checkpoint.is_file() or session.state.update_idx != start_update:
            final_checkpoint = checkpoints / f"update-{session.state.update_idx:06d}.pt"
            session.save_checkpoint(
                final_checkpoint,
                git_sha=metadata.git_sha,
                git_dirty=metadata.git_dirty,
            )
        scientific_final = session.state.update_idx >= config.total_updates or (
            config.duration_seconds is not None
            and session.state.elapsed_training_seconds >= config.duration_seconds
        )
        if scientific_final:
            final_evaluation_path = _run_evaluation(
                session=session,
                logger=logger,
                history=history,
                output_directory=output_directory,
                games=config.eval_games,
                kind="final",
            )
            evaluation_path = final_evaluation_path
            logger.assert_complete()
            session.save_checkpoint(
                final_checkpoint,
                git_sha=metadata.git_sha,
                git_dirty=metadata.git_dirty,
            )
            resolved_config = output_directory / "config.resolved.yaml"
            files = [
                final_checkpoint,
                final_checkpoint.with_suffix(f"{final_checkpoint.suffix}.sha256"),
                resolved_config,
                metrics_path,
                final_evaluation_path,
            ]
            checklist = repository / "docs" / "PPO_CHECKLIST.md"
            if checklist.is_file():
                files.append(checklist)
            diff = _source_diff(repository, output_directory)
            if diff is not None:
                files.append(diff)
            if recovery_context is not None:
                files.extend(recovery_context.artifact_files)
            artifact_name = (
                f"checkpoint-{config.experiment_id}-update-{session.state.update_idx:06d}"
            )
            artifact_metadata: dict[str, object] = {
                "experiment_id": config.experiment_id,
                "seed": config.seed,
                "update_idx": session.state.update_idx,
                "global_step": session.state.global_step,
                "git_sha": metadata.git_sha,
                "git_dirty": metadata.git_dirty,
            }
            if recovery_context is not None:
                artifact_metadata.update(
                    {
                        "recovery_manifest_sha256": recovery_context.manifest_sha256,
                        "recovery_source_commit": recovery_context.source_commit,
                        "recovery_checkpoint_sha256": (recovery_context.source_checkpoint_sha256),
                    }
                )
            logger.log_artifact(
                name=artifact_name,
                artifact_type="model",
                files=tuple(files),
                metadata=artifact_metadata,
            )
        status = "completed"
    except BaseException as error:
        runtime_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if logger is not None:
                logger.finish(exit_code=0 if status == "completed" else 1)
        except BaseException as error:
            status = "failed"
            runtime_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            terminal_runtime = finish_runtime_state(
                runtime_state,
                status="COMPLETED" if status == "completed" else "FAILED",
                latest_error=runtime_error,
            )
            write_runtime_state(runtime_path, terminal_runtime)

    wall_seconds = time.perf_counter() - wall_started
    if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise RuntimeError("wall clock produced an invalid elapsed duration")
    manifest = {
        "schema": "CHECKERS_TRAINING_RUN_1",
        "status": status,
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "phase": config.phase,
        "stage": config.stage,
        "arm": config.arm,
        "device": config.device,
        "deterministic": config.deterministic,
        "git_sha": metadata.git_sha,
        "git_dirty": metadata.git_dirty,
        "start_update": start_update,
        "end_update": session.state.update_idx,
        "global_step": session.state.global_step,
        "elapsed_training_seconds": session.state.elapsed_training_seconds,
        "wall_seconds": wall_seconds,
        "wandb_run_id": session.state.wandb_run_id,
        "logging_step": session.state.logging_step,
        "checkpoint": str(final_checkpoint),
        "evaluation": None if evaluation_path is None else str(evaluation_path),
        "evaluation_games": config.eval_games if scientific_final else None,
        "metrics_history": str(metrics_path),
        "wandb_artifact": artifact_name,
        "resume_from": None if resume_path is None else str(resume_path),
        "recovery": (
            None
            if recovery_context is None
            else {
                "manifest": str(recovery_context.manifest_path),
                "manifest_sha256": recovery_context.manifest_sha256,
                "source_commit": recovery_context.source_commit,
                "recovery_commit": recovery_context.recovery_commit,
                "checkpoint_update": recovery_context.checkpoint_update_idx,
                "checkpoint_sha256": recovery_context.source_checkpoint_sha256,
            }
        ),
    }
    manifest_path = output_directory / f"manifest-{session.state.update_idx:06d}.json"
    atomic_write_bytes(manifest_path, _manifest_bytes(manifest))
    return TrainingRunResult(
        start_update=start_update,
        end_update=session.state.update_idx,
        checkpoint_path=final_checkpoint,
        manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        metrics_path=metrics_path,
        wandb_run_id=session.state.wandb_run_id,
        logging_step=session.state.logging_step,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the stable training CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkers-phase7.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/checkers-ppo"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options, execute training, and print one machine-readable result."""

    arguments = build_parser().parse_args(argv)
    result = run_training(
        config_path=arguments.config,
        output_directory=arguments.output_dir,
        resume_path=arguments.resume,
        max_updates=arguments.max_updates,
    )
    print(
        json.dumps(
            {
                "start_update": result.start_update,
                "end_update": result.end_update,
                "checkpoint": str(result.checkpoint_path),
                "manifest": str(result.manifest_path),
                "wandb_run_id": result.wandb_run_id,
                "logging_step": result.logging_step,
            },
            sort_keys=True,
        )
    )
    return 0
