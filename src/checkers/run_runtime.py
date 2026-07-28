"""Atomic process-lifecycle state for read-only run monitoring."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil

from checkers.eval.baseline_run import atomic_write_bytes

RUNTIME_SCHEMA = "CHECKERS_TRAINING_RUNTIME_1"
RUNTIME_STATUSES = frozenset({"RUNNING", "COMPLETED", "FAILED"})


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Small durable lifecycle record; metrics/checkpoints remain authoritative progress."""

    schema: str
    status: str
    pid: int
    started_at: str
    updated_at: str
    start_update: int
    experiment_id: str
    seed: int
    git_sha: str
    run_id: str | None
    resume_from: str | None
    latest_warning: str | None
    latest_error: str | None
    process_start_ticks: int | None = None


def read_process_start_ticks(pid: int) -> int | None:
    """Return Linux's boot-relative process start token, or None when unavailable."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("pid must be a positive integer")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields_after_command = stat[command_end + 1 :].split()
    start_time_index = 19  # field 22, after removing PID and parenthesized command fields
    if len(fields_after_command) <= start_time_index:
        return None
    try:
        start_ticks = int(fields_after_command[start_time_index])
    except ValueError:
        return None
    return start_ticks if start_ticks > 0 else None


def new_runtime_state(  # noqa: PLR0913
    *,
    start_update: int,
    experiment_id: str,
    seed: int,
    git_sha: str,
    run_id: str | None,
    resume_from: Path | None,
) -> RuntimeState:
    """Return a RUNNING lifecycle record for the current process."""

    if isinstance(start_update, bool) or not isinstance(start_update, int) or start_update < 0:
        raise ValueError("start_update must be a non-negative integer")
    if resume_from is not None and not isinstance(resume_from, Path):
        raise TypeError("resume_from must be a Path or None")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty text")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("git_sha must be non-empty text")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("run_id must be non-empty text or None")
    pid = os.getpid()
    now = datetime.now(UTC).isoformat()
    started_at = datetime.fromtimestamp(psutil.Process(pid).create_time(), UTC).isoformat()
    process_start_ticks = read_process_start_ticks(pid)
    if process_start_ticks is None:
        raise RuntimeError("operating-system process start token is unavailable")
    return RuntimeState(
        schema=RUNTIME_SCHEMA,
        status="RUNNING",
        pid=pid,
        started_at=started_at,
        updated_at=now,
        start_update=start_update,
        experiment_id=experiment_id,
        seed=seed,
        git_sha=git_sha,
        run_id=run_id,
        resume_from=None if resume_from is None else str(resume_from),
        latest_warning=None,
        latest_error=None,
        process_start_ticks=process_start_ticks,
    )


def finish_runtime_state(
    state: RuntimeState,
    *,
    status: str,
    latest_error: str | None = None,
) -> RuntimeState:
    """Return a terminal lifecycle record without changing its process identity."""

    if not isinstance(state, RuntimeState):
        raise TypeError("state must be a RuntimeState")
    if status not in RUNTIME_STATUSES - {"RUNNING"}:
        raise ValueError("terminal runtime status must be COMPLETED or FAILED")
    if latest_error is not None and not isinstance(latest_error, str):
        raise TypeError("latest_error must be text or None")
    return RuntimeState(
        schema=state.schema,
        status=status,
        pid=state.pid,
        started_at=state.started_at,
        updated_at=datetime.now(UTC).isoformat(),
        start_update=state.start_update,
        experiment_id=state.experiment_id,
        seed=state.seed,
        git_sha=state.git_sha,
        run_id=state.run_id,
        resume_from=state.resume_from,
        latest_warning=state.latest_warning,
        latest_error=latest_error,
        process_start_ticks=state.process_start_ticks,
    )


def attach_runtime_run_id(state: RuntimeState, *, run_id: str) -> RuntimeState:
    """Return the RUNNING record with the stable W&B identity attached."""

    if not isinstance(state, RuntimeState):
        raise TypeError("state must be a RuntimeState")
    if state.status != "RUNNING":
        raise ValueError("run identity can only be attached while RUNNING")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty text")
    return RuntimeState(
        schema=state.schema,
        status=state.status,
        pid=state.pid,
        started_at=state.started_at,
        updated_at=datetime.now(UTC).isoformat(),
        start_update=state.start_update,
        experiment_id=state.experiment_id,
        seed=state.seed,
        git_sha=state.git_sha,
        run_id=run_id,
        resume_from=state.resume_from,
        latest_warning=state.latest_warning,
        latest_error=state.latest_error,
        process_start_ticks=state.process_start_ticks,
    )


def write_runtime_state(path: Path, state: RuntimeState) -> None:
    """Atomically replace the monitor lifecycle record."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(state, RuntimeState):
        raise TypeError("state must be a RuntimeState")
    payload = (json.dumps(asdict(state), sort_keys=True, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def read_runtime_state(path: Path) -> RuntimeState | None:
    """Read a lifecycle record, returning None only when it does not exist."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("runtime state is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("runtime state root must be a mapping")
    try:
        state = RuntimeState(**value)
    except TypeError as error:
        raise ValueError("runtime state fields are invalid") from error
    if state.schema != RUNTIME_SCHEMA or state.status not in RUNTIME_STATUSES:
        raise ValueError("runtime state schema or status is invalid")
    if state.pid < 1 or state.start_update < 0:
        raise ValueError("runtime state counters are invalid")
    if state.process_start_ticks is not None and state.process_start_ticks < 1:
        raise ValueError("runtime process start token is invalid")
    return state
