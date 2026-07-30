"""Offline orchestration tests for the resumable baseline CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from checkers.eval.arena import STREAMS_PER_GAME, MatchResult, play_balanced_match
from checkers.eval.baseline_eval import (
    BaselineConfig,
    BaselineMatchSummary,
    agent_spec,
    load_baseline_config,
    summarize_match,
)
from checkers.eval.baseline_run import RunIdentity
from checkers.rules.state import PlayerId, State
from scripts import evaluate_baselines

GIT_SHA = "a" * 40
SMALL_GAMES = 2
EXPECTED_EVENTS = 8


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _small_match(pair: tuple[str, str], seed: int) -> MatchResult:
    return play_balanced_match(
        first=agent_spec(pair[0], max_plies=512),
        second=agent_spec(pair[1], max_plies=512),
        games=SMALL_GAMES,
        seed=seed,
        initial_state=State(
            men=(_mask(9), _mask(14)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        ),
    )


def test_cli_orchestration_writes_loadable_report_archive_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = tmp_path / "games.json.gz"
    report_path = tmp_path / "report.json"
    log_path = tmp_path / "progress.jsonl"

    def fake_load_or_play(
        *,
        config: BaselineConfig,
        identity: RunIdentity,
        checkpoint_dir: Path,
        index: int,
        pair: tuple[str, str],
    ) -> tuple[MatchResult, BaselineMatchSummary, str]:
        del identity, checkpoint_dir
        match = _small_match(pair, evaluate_baselines._comparison_seed(config, index))
        return match, summarize_match(match, elapsed_seconds=0.01), "simulated"

    monkeypatch.setattr(evaluate_baselines, "_clean_git_commit", lambda: GIT_SHA)
    monkeypatch.setattr(evaluate_baselines, "_load_or_play", fake_load_or_play)
    monkeypatch.setattr(
        evaluate_baselines,
        "_gpu_snapshot",
        lambda: {"status": "TEST_DOUBLE"},
    )
    arguments = evaluate_baselines._parser().parse_args(
        [
            "--config",
            "configs/checkers-baselines-v1.yaml",
            "--contract",
            "docs/experiment-contract.md",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--raw-output",
            str(raw_path),
            "--report-output",
            str(report_path),
            "--progress-log",
            str(log_path),
        ]
    )

    assert evaluate_baselines.run(arguments) == 0

    report = cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
    identity = cast(dict[str, object], report["identity"])
    gate = cast(dict[str, object], report["gate_5"])
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert raw_path.read_bytes().startswith(b"\x1f\x8b")
    assert identity["git_commit"] == GIT_SHA
    assert gate["technical_pass"] is True
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert len(events) == EXPECTED_EVENTS


def test_cli_helpers_reject_dirty_tree_bad_json_and_bad_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_baselines, "_git_output", lambda *_args: "dirty")
    with pytest.raises(RuntimeError, match="clean Git"):
        evaluate_baselines._clean_git_commit()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        evaluate_baselines._load_json(invalid)
    with pytest.raises(TypeError, match="mapping"):
        evaluate_baselines._report_gate_passed([])
    with pytest.raises(ValueError, match="gate_5"):
        evaluate_baselines._report_gate_passed({})
    with pytest.raises(TypeError, match="bool"):
        evaluate_baselines._report_gate_passed({"gate_5": {"technical_pass": 1}})


def test_checkpoint_names_are_stable_and_shell_safe() -> None:
    path = evaluate_baselines._checkpoint_path(
        Path("runs"),
        index=5,
        first="minimax(2)",
        second="minimax(1)",
    )
    assert path == Path("runs/05-minimax-2-vs-minimax-1.json")


def test_comparison_seed_blocks_are_disjoint_across_full_experiment() -> None:
    config = load_baseline_config(
        Path("configs/checkers-baselines-v1.yaml").read_text(encoding="utf-8")
    )
    roots = tuple(
        evaluate_baselines._comparison_seed(config, index)
        for index in range(len(config.comparisons))
    )
    stride = config.games_per_match * STREAMS_PER_GAME

    assert roots == tuple(config.seed + index * stride for index in range(len(roots)))
    assert all(left + stride <= right for left, right in zip(roots[:-1], roots[1:], strict=True))
    with pytest.raises(ValueError, match="outside"):
        evaluate_baselines._comparison_seed(config, -1)
    with pytest.raises(ValueError, match="outside"):
        evaluate_baselines._comparison_seed(config, len(config.comparisons))
