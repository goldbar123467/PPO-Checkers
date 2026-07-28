"""Append-only local metric history integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkers.metric_history import MetricHistoryWriter
from checkers.trainer_state import TrainerState

GLOBAL_STEP = 24


def test_metric_history_appends_monotonic_resume_safe_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    state = TrainerState(update_idx=3, global_step=GLOBAL_STEP, logging_step=0)
    writer = MetricHistoryWriter(path=path, next_logging_step=0)
    writer.append(
        kind="training",
        metrics={"train/policy_loss": 0.25},
        state=state,
        logging_step=0,
    )
    state.logging_step = 1
    resumed = MetricHistoryWriter(path=path, next_logging_step=1)
    resumed.append(
        kind="final_evaluation",
        metrics={"eval/vs_random": 0.9},
        state=state,
        logging_step=1,
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [record["logging_step"] for record in records] == [0, 1]
    assert [record["kind"] for record in records] == ["training", "final_evaluation"]
    assert records[0]["global_step"] == GLOBAL_STEP
    assert records[1]["metrics"] == {"eval/vs_random": 0.9}
    with pytest.raises(ValueError, match="next logging step"):
        MetricHistoryWriter(path=path, next_logging_step=1)
