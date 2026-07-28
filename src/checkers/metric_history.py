"""Fsync-backed append-only JSONL mirror of scalar offline run metrics."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from checkers.trainer_state import TrainerState

SCHEMA = "CHECKERS_METRIC_HISTORY_1"


def _last_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError("metric history ends with invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("metric history record must be a mapping")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written < 1:
            raise OSError("metric history append made no progress")
        offset += written


class MetricHistoryWriter:
    """Validate the resume boundary and append one durable scalar record at a time."""

    def __init__(self, *, path: Path, next_logging_step: int) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        if isinstance(next_logging_step, bool) or not isinstance(next_logging_step, int):
            raise TypeError("next_logging_step must be an integer")
        if next_logging_step < 0:
            raise ValueError("next_logging_step must be non-negative")
        last = _last_record(path)
        if last is None:
            expected = 0
        else:
            previous_step = last.get("logging_step")
            if isinstance(previous_step, bool) or not isinstance(previous_step, int):
                raise ValueError("metric history logging_step is invalid")
            expected = previous_step + 1
        if expected != next_logging_step:
            raise ValueError("metric history and next logging step disagree")
        self._path = path
        self._next_logging_step = next_logging_step

    def append(
        self,
        *,
        kind: str,
        metrics: Mapping[str, float],
        state: TrainerState,
        logging_step: int,
    ) -> None:
        """Append and fsync one finite scalar record at the exact trainer-owned step."""

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be non-empty text")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("metrics must be a non-empty mapping")
        if not isinstance(state, TrainerState):
            raise TypeError("state must be a TrainerState")
        if isinstance(logging_step, bool) or not isinstance(logging_step, int):
            raise TypeError("logging_step must be an integer")
        if logging_step != self._next_logging_step:
            raise ValueError("metric-history logging step is not monotonic")
        record = {
            "schema": SCHEMA,
            "logging_step": logging_step,
            "kind": kind.strip(),
            "update_idx": state.update_idx,
            "global_step": state.global_step,
            "elapsed_training_seconds": state.elapsed_training_seconds,
            "metrics": dict(metrics),
        }
        payload = (
            json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._next_logging_step += 1
