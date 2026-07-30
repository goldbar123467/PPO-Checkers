"""Atomic runtime lifecycle validation tests."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import psutil
import pytest

from checkers.run_runtime import (
    RUNTIME_SCHEMA,
    RuntimeState,
    attach_runtime_run_id,
    finish_runtime_state,
    new_runtime_state,
    read_process_start_ticks,
    read_runtime_state,
    write_runtime_state,
)


def _state() -> RuntimeState:
    return new_runtime_state(
        start_update=3,
        experiment_id="runtime-unit",
        seed=5,
        git_sha="abcdef",
        run_id=None,
        resume_from=Path("checkpoint.pt"),
    )


def test_runtime_identity_uses_the_os_process_creation_time() -> None:
    state = _state()

    assert state.pid == os.getpid()
    assert state.process_start_ticks == read_process_start_ticks(state.pid)
    assert datetime.fromisoformat(state.started_at).timestamp() == pytest.approx(
        psutil.Process(state.pid).create_time(),
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"start_update": -1}, "start_update"),
        ({"resume_from": "bad"}, "resume_from"),
        ({"experiment_id": ""}, "experiment_id"),
        ({"seed": -1}, "seed"),
        ({"git_sha": ""}, "git_sha"),
        ({"run_id": ""}, "run_id"),
    ],
)
def test_new_runtime_state_rejects_invalid_identity(
    overrides: dict[str, object], error: str
) -> None:
    arguments: dict[str, object] = {
        "start_update": 0,
        "experiment_id": "runtime-unit",
        "seed": 1,
        "git_sha": "abc",
        "run_id": None,
        "resume_from": None,
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError), match=error):
        new_runtime_state(**arguments)  # type: ignore[arg-type]


def test_runtime_transitions_validate_types_and_statuses() -> None:
    state = _state()
    attached = attach_runtime_run_id(state, run_id="stable-id")
    failed = finish_runtime_state(attached, status="FAILED", latest_error="boom")

    assert attached.run_id == "stable-id"
    assert failed.status == "FAILED"
    assert failed.latest_error == "boom"
    with pytest.raises(TypeError, match="RuntimeState"):
        finish_runtime_state(cast(RuntimeState, "bad"), status="FAILED")
    with pytest.raises(ValueError, match="terminal runtime status"):
        finish_runtime_state(state, status="RUNNING")
    with pytest.raises(TypeError, match="latest_error"):
        finish_runtime_state(state, status="FAILED", latest_error=cast(str, 1))
    with pytest.raises(TypeError, match="RuntimeState"):
        attach_runtime_run_id(cast(RuntimeState, "bad"), run_id="id")
    with pytest.raises(ValueError, match="while RUNNING"):
        attach_runtime_run_id(failed, run_id="id")
    with pytest.raises(ValueError, match="run_id"):
        attach_runtime_run_id(state, run_id="")


def test_runtime_round_trip_and_reader_failures(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    state = _state()

    assert read_runtime_state(path) is None
    write_runtime_state(path, state)
    assert read_runtime_state(path) == state
    with pytest.raises(TypeError, match="path"):
        write_runtime_state(cast(Path, "bad"), state)
    with pytest.raises(TypeError, match="RuntimeState"):
        write_runtime_state(path, cast(RuntimeState, "bad"))
    with pytest.raises(TypeError, match="path"):
        read_runtime_state(cast(Path, "bad"))

    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        read_runtime_state(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        read_runtime_state(path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        read_runtime_state(path)
    invalid_schema = {**asdict(state), "schema": "BAD"}
    path.write_text(json.dumps(invalid_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="schema or status"):
        read_runtime_state(path)
    invalid_counter = asdict(replace(state, pid=0))
    invalid_counter["schema"] = RUNTIME_SCHEMA
    path.write_text(json.dumps(invalid_counter), encoding="utf-8")
    with pytest.raises(ValueError, match="counters"):
        read_runtime_state(path)
    invalid_token = asdict(replace(state, process_start_ticks=0))
    path.write_text(json.dumps(invalid_token), encoding="utf-8")
    with pytest.raises(ValueError, match="start token"):
        read_runtime_state(path)


@pytest.mark.parametrize("pid", [0, -1, True])
def test_process_start_ticks_rejects_invalid_pid(pid: int) -> None:
    with pytest.raises(ValueError, match="pid"):
        read_process_start_ticks(pid)


def test_process_start_ticks_returns_none_for_missing_process() -> None:
    assert read_process_start_ticks(2**31 - 1) is None
