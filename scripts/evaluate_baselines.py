#!/usr/bin/env python3
"""Run or resume the frozen, powered Phase 5 baseline round robin."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, cast

from checkers.eval.arena import STREAMS_PER_GAME, MatchResult, play_balanced_match
from checkers.eval.baseline_eval import (
    BaselineConfig,
    BaselineMatchSummary,
    agent_spec,
    build_evaluation_report,
    build_tactical_report,
    load_baseline_config,
    summarize_match,
)
from checkers.eval.baseline_run import (
    RunIdentity,
    atomic_write_bytes,
    build_checkpoint,
    build_raw_archive,
    parse_checkpoint,
    parse_raw_archive,
    sha256_bytes,
)
from checkers.rules.state import State

DEFAULT_CONFIG = Path("configs/checkers-baselines-v1.yaml")
DEFAULT_CHECKPOINT_DIR = Path("runs/metadata/phase5-baselines-v1")
DEFAULT_RAW_OUTPUT = Path("reports/phase5_baseline_games_v1.json.gz")
DEFAULT_REPORT_OUTPUT = Path("reports/phase5_baseline_report_v1.json")
PACKAGE_NAMES = ("python", "ppo-checkers", "numpy", "gymnasium", "pyyaml")
DEFAULT_CONTRACT = Path("docs/experiment-contract.md")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load valid JSON checkpoint {path}") from error


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_git_commit() -> str:
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("baseline evaluation requires a clean Git worktree")
    commit = _git_output("rev-parse", "HEAD")
    return RunIdentity(
        experiment_id="validation-only",
        git_commit=commit,
        config_sha256="0" * 64,
        goal_sha256="0" * 64,
    ).git_commit


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_path(
    directory: Path,
    *,
    index: int,
    first: str,
    second: str,
) -> Path:
    def slug(name: str) -> str:
        return name.replace("(", "-").replace(")", "").replace("/", "-")

    return directory / f"{index:02d}-{slug(first)}-vs-{slug(second)}.json"


def _comparison_seed(config: BaselineConfig, index: int) -> int:
    if not 0 <= index < len(config.comparisons):
        raise ValueError("comparison index is outside the configured schedule")
    return config.seed + index * config.games_per_match * STREAMS_PER_GAME


def _validate_configured_match(
    match: MatchResult,
    *,
    config: BaselineConfig,
    pair: tuple[str, str],
    seed: int,
) -> None:
    if (match.first_agent, match.second_agent) != pair:
        raise ValueError("match pair does not match configuration")
    if match.seed != seed:
        raise ValueError("match seed does not match configuration")
    if match.games != config.games_per_match:
        raise ValueError("match game count does not match configuration")
    if match.initial_state != State.initial():
        raise ValueError("match must begin from the standard initial state")
    if match.max_plies != config.max_plies:
        raise ValueError("match max_plies does not match configuration")
    if match.repetition_draws is not config.repetition_draws:
        raise ValueError("match repetition setting does not match configuration")
    if not math.isclose(match.confidence, config.confidence, abs_tol=0.0):
        raise ValueError("match confidence does not match configuration")


def _write_checkpoint(
    path: Path,
    *,
    identity: RunIdentity,
    index: int,
    match: MatchResult,
    summary: BaselineMatchSummary,
) -> None:
    record = build_checkpoint(
        identity=identity,
        comparison_index=index,
        match=match,
        summary=summary,
    )
    atomic_write_bytes(path, _canonical_json_bytes(record))


def _load_or_play(  # noqa: PLR0913
    *,
    config: BaselineConfig,
    identity: RunIdentity,
    checkpoint_dir: Path,
    index: int,
    pair: tuple[str, str],
) -> tuple[MatchResult, BaselineMatchSummary, str]:
    first, second = pair
    seed = _comparison_seed(config, index)
    checkpoint = _checkpoint_path(
        checkpoint_dir,
        index=index,
        first=first,
        second=second,
    )
    if checkpoint.is_file():
        match, summary = parse_checkpoint(
            _load_json(checkpoint),
            identity=identity,
            comparison_index=index,
            expected_pair=pair,
            expected_seed=seed,
        )
        _validate_configured_match(match, config=config, pair=pair, seed=seed)
        return match, summary, "resumed"

    started = time.perf_counter()
    match = play_balanced_match(
        first=agent_spec(first, max_plies=config.max_plies),
        second=agent_spec(second, max_plies=config.max_plies),
        games=config.games_per_match,
        seed=seed,
        max_plies=config.max_plies,
        repetition_draws=config.repetition_draws,
        confidence=config.confidence,
    )
    elapsed = time.perf_counter() - started
    _validate_configured_match(match, config=config, pair=pair, seed=seed)
    summary = summarize_match(match, elapsed_seconds=elapsed)
    _write_checkpoint(
        checkpoint,
        identity=identity,
        index=index,
        match=match,
        summary=summary,
    )
    restored_match, restored_summary = parse_checkpoint(
        _load_json(checkpoint),
        identity=identity,
        comparison_index=index,
        expected_pair=pair,
        expected_seed=seed,
    )
    if restored_match != match or restored_summary != summary:
        raise RuntimeError("new checkpoint failed immediate load validation")
    return match, summary, "completed"


def _gpu_snapshot() -> dict[str, object]:
    command = (
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"status": "UNAVAILABLE", "error_type": type(error).__name__}
    return {"status": "AVAILABLE", "rows": completed.stdout.strip().splitlines()}


def _dependencies() -> dict[str, object]:
    versions: dict[str, object] = {"python": platform.python_version()}
    for package in PACKAGE_NAMES:
        if package == "python":
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _emit(record: dict[str, object], log: TextIO | None) -> None:
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    if log is not None:
        log.write(line + "\n")
        log.flush()


def _report_gate_passed(report: object) -> bool:
    if not isinstance(report, dict):
        raise TypeError("reloaded report must be a mapping")
    root = cast(dict[object, object], report)
    gate = root.get("gate_5")
    if not isinstance(gate, dict):
        raise ValueError("reloaded report must contain gate_5")
    value = cast(dict[object, object], gate).get("technical_pass")
    if not isinstance(value, bool):
        raise TypeError("gate_5.technical_pass must be bool")
    return value


def run(arguments: argparse.Namespace) -> int:
    """Execute or resume the complete baseline evaluation described by parsed arguments.

    Args:
        arguments: Paths parsed by this module's CLI parser.

    Returns:
        Zero only after saved replay and report artifacts reload and Gate 5 passes.

    Raises:
        OSError: If required inputs or output paths cannot be read or written.
        RuntimeError: If Git is dirty, persistence validation fails, or the gate is red.
        TypeError: If a loaded artifact has an invalid runtime type.
        ValueError: If configuration, identity, schedule, or artifact content disagrees.
    """

    config_path = cast(Path, arguments.config)
    checkpoint_dir = cast(Path, arguments.checkpoint_dir)
    raw_output = cast(Path, arguments.raw_output)
    report_output = cast(Path, arguments.report_output)
    progress_log = cast(Path | None, arguments.progress_log)
    contract_path = cast(Path, arguments.contract)

    git_commit = _clean_git_commit()
    config_text = config_path.read_text(encoding="utf-8")
    config = load_baseline_config(config_text)
    identity = RunIdentity(
        experiment_id=config.experiment_id,
        git_commit=git_commit,
        config_sha256=hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        goal_sha256=_file_sha256(contract_path),
    )
    log_handle = None if progress_log is None else progress_log.open("w", encoding="utf-8")
    try:
        gpu_before = _gpu_snapshot()
        _emit(
            {
                "event": "run_started",
                "experiment_id": config.experiment_id,
                "git_commit": git_commit,
                "games_per_match": config.games_per_match,
                "comparisons": len(config.comparisons),
                "gpu": gpu_before,
            },
            log_handle,
        )
        matches: list[MatchResult] = []
        summaries: list[BaselineMatchSummary] = []
        for index, pair in enumerate(config.comparisons):
            match, summary, status = _load_or_play(
                config=config,
                identity=identity,
                checkpoint_dir=checkpoint_dir,
                index=index,
                pair=pair,
            )
            matches.append(match)
            summaries.append(summary)
            _emit(
                {
                    "event": "comparison_finished",
                    "index": index,
                    "status": status,
                    "first_agent": match.first_agent,
                    "second_agent": match.second_agent,
                    "games": match.games,
                    "wins": match.wins,
                    "draws": match.draws,
                    "losses": match.losses,
                    "score": match.score.score,
                    "ci_low": match.score.low,
                    "ci_high": match.score.high,
                    "elapsed_seconds": summary.elapsed_seconds,
                    "gpu": _gpu_snapshot(),
                },
                log_handle,
            )

        checked_matches = tuple(matches)
        raw_bytes = build_raw_archive(identity=identity, matches=checked_matches)
        atomic_write_bytes(raw_output, raw_bytes)
        raw_sha = sha256_bytes(raw_bytes)
        if parse_raw_archive(raw_output.read_bytes(), identity=identity) != checked_matches:
            raise RuntimeError("saved raw archive failed load validation")
        tactical = build_tactical_report(seed=config.seed, depths=config.tactical_depths)
        report = build_evaluation_report(
            config=config,
            matches=checked_matches,
            summaries=tuple(summaries),
            tactical=tactical,
            git_commit=git_commit,
            config_sha256=identity.config_sha256,
            goal_sha256=identity.goal_sha256,
            raw_games_sha256=raw_sha,
            hardware={
                "execution_device": "cpu",
                "execution_reason": "fixed search policies contain no tensor/GPU backend",
                "platform": platform.platform(),
                "machine": platform.machine(),
                "gpu_before": gpu_before,
                "gpu_after": _gpu_snapshot(),
            },
            dependencies=_dependencies(),
        )
        report_bytes = _canonical_json_bytes(report)
        atomic_write_bytes(report_output, report_bytes)
        reloaded: object = json.loads(report_output.read_text(encoding="utf-8"))
        passed = _report_gate_passed(reloaded)
        _emit(
            {
                "event": "run_finished",
                "technical_pass": passed,
                "raw_games_sha256": raw_sha,
                "raw_output": str(raw_output),
                "report_output": str(report_output),
                "report_sha256": sha256_bytes(report_bytes),
            },
            log_handle,
        )
        if not passed:
            raise RuntimeError("Phase 5 technical gate failed; preserved report for diagnosis")
        return 0
    finally:
        if log_handle is not None:
            log_handle.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--contract",
        "--goal",
        dest="contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="public experiment contract to hash into the evaluation identity",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--progress-log", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the deterministic baseline experiment.

    Args:
        argv: Optional explicit CLI argument sequence.

    Returns:
        Process-style zero status after complete validation.

    Raises:
        OSError: If required files cannot be read or written.
        RuntimeError: If the run cannot safely proceed or its technical gate fails.
        TypeError: If a persisted record has an invalid runtime type.
        ValueError: If configuration or persisted evidence is inconsistent.
    """

    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
