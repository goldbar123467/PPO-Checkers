"""Fail-open W&B observability, monotonic logging, and credential auditing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
import wandb

from checkers.config import WANDB_MODES, RunConfig
from checkers.metrics import REQUIRED_METRIC_KEYS
from checkers.trainer_state import TrainerState

PROJECT_NAME = "checkers-ppo"
PROVENANCE_PREFIX = "provenance/"
PAYOFF_TABLE_COLUMNS = (
    "row_agent",
    "column_agent",
    "wins",
    "draws",
    "losses",
    "games",
    "score",
)
GAME_TABLE_COLUMNS = (
    "match",
    "game_index",
    "perspective_agent",
    "perspective_result",
    "red_agent",
    "white_agent",
    "winner",
    "termination_reason",
    "steps",
    "completed_moves",
    "moves",
    "actions",
    "red_seed",
    "white_seed",
    "environment_seed",
)
CREDENTIAL_FILENAMES = frozenset(
    {
        ".env",
        ".secrets",
        ".netrc",
        "kaggle.json",
        "wandb_api_key",
    }
)
WANDB_FAILURE_SCHEMA = "CHECKERS_WANDB_FAILURE_1"
_CREDENTIAL_PATTERNS = (
    (
        "40-character API key",
        re.compile(rb"(?i)(?:api[_-]?key|wandb_api_key)\s*[:=]\s*['\"]?[a-f0-9]{40}"),
    ),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


class _Run(Protocol):
    id: str

    @property
    def summary(self) -> MutableMapping[str, object]: ...

    def log(self, data: dict[str, object], *, step: int, commit: bool) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...

    def log_artifact(self, artifact: wandb.Artifact) -> object: ...


@contextmanager
def _preserve_global_rngs() -> Iterator[None]:
    """Prevent synchronous observer code from consuming trainer-global RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        else ()
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))


def _append_failure(path: Path | None, *, operation: str, error: Exception) -> None:
    """Append a secret-free W&B failure record, with stderr as the last fallback."""

    record = {
        "schema": WANDB_FAILURE_SCHEMA,
        "operation": operation,
        "error_type": type(error).__name__,
    }
    rendered = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        if path is None:
            raise OSError("no failure path configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            payload = rendered.encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written < 1:
                    raise OSError("W&B failure append made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        print(
            f"wandb_failure operation={operation} error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Immutable source, dependency, host, device, and determinism provenance."""

    git_sha: str
    git_dirty: bool
    hostname: str
    python_version: str
    torch_version: str
    numpy_version: str
    gymnasium_version: str
    wandb_version: str
    device_name: str
    deterministic: bool

    def as_summary(self) -> dict[str, object]:
        """Return namespaced values without mutating the verbatim run configuration."""

        return {f"{PROVENANCE_PREFIX}{key}": value for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CredentialFinding:
    """A path and non-secret reason from a repository credential scan."""

    path: Path
    reason: str


def _table(
    rows: tuple[dict[str, object], ...],
    *,
    columns: tuple[str, ...],
    name: str,
) -> wandb.Table:
    if not isinstance(rows, tuple):
        raise TypeError(f"{name} rows must be a tuple")
    if not rows:
        raise ValueError(f"{name} rows must not be empty")
    data: list[list[object]] = []
    expected = set(columns)
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{name} rows must contain mappings")
        if set(row) != expected:
            raise ValueError(f"{name} row fields are invalid")
        data.append([row[column] for column in columns])
    return wandb.Table(columns=list(columns), data=data)


def payoff_matrix_table(rows: tuple[dict[str, object], ...]) -> wandb.Table:
    """Build a literal W/D/L payoff table without reducing its cells to scalar Elo."""

    return _table(rows, columns=PAYOFF_TABLE_COLUMNS, name="payoff matrix")


def game_table(rows: tuple[dict[str, object], ...]) -> wandb.Table:
    """Build a replayable human-readable table of ACF moves and action IDs."""

    return _table(rows, columns=GAME_TABLE_COLUMNS, name="game")


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_run_metadata(*, config: RunConfig, repository: Path) -> RunMetadata:
    """Read local-only immutable provenance for one run."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(repository, Path):
        raise TypeError("repository must be a Path")
    git_sha = _git_output(repository, "rev-parse", "HEAD")
    git_dirty = bool(_git_output(repository, "status", "--porcelain"))
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if config.device == "cuda"
        else "cpu"
    )
    return RunMetadata(
        git_sha=git_sha,
        git_dirty=git_dirty,
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        gymnasium_version=version("gymnasium"),
        wandb_version=version("wandb"),
        device_name=device_name,
        deterministic=config.deterministic,
    )


class WandbLogger:
    """Fail-open run wrapper enforcing explicit, monotonic, resumable log steps."""

    def __init__(
        self,
        *,
        run: _Run | None,
        initial_logging_step: int,
        failure_path: Path | None = None,
    ) -> None:
        if run is not None and (not hasattr(run, "log") or not hasattr(run, "finish")):
            raise TypeError("run must provide log and finish methods")
        if isinstance(initial_logging_step, bool) or not isinstance(initial_logging_step, int):
            raise TypeError("initial_logging_step must be an integer")
        if initial_logging_step < 0:
            raise ValueError("initial_logging_step must be non-negative")
        if failure_path is not None and not isinstance(failure_path, Path):
            raise TypeError("failure_path must be a Path or None")
        self._run = run
        self._failure_path = failure_path
        self._last_logging_step = initial_logging_step - 1
        self._observed_metric_keys: set[str] = set()

    @property
    def observed_metric_keys(self) -> frozenset[str]:
        """Return required metric keys observed at least once in this logger session."""

        return frozenset(self._observed_metric_keys)

    def log(self, metrics: Mapping[str, object], *, state: TrainerState) -> None:
        """Attempt one cloud log and always advance the trainer-owned observer step."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if not isinstance(state, TrainerState):
            raise TypeError("state must be a TrainerState")
        payload = dict(metrics)
        if not payload:
            raise ValueError("metrics must not be empty")
        if any(not isinstance(key, str) or not key for key in payload):
            raise TypeError("metric names must be non-empty strings")
        if state.logging_step <= self._last_logging_step:
            raise ValueError("logging step must be strictly monotonic")
        if self._run is not None:
            try:
                with _preserve_global_rngs():
                    self._run.log(payload, step=state.logging_step, commit=True)
            except Exception as error:  # W&B is convenience; local history is authoritative.
                _append_failure(self._failure_path, operation="log", error=error)
        self._last_logging_step = state.logging_step
        self._observed_metric_keys.update(REQUIRED_METRIC_KEYS.intersection(payload))
        state.advance_logging_step()

    def update_summary(self, values: Mapping[str, object]) -> None:
        """Attempt a W&B summary update without allowing SDK failure to propagate."""

        if not isinstance(values, Mapping):
            raise TypeError("summary values must be a mapping")
        if any(not isinstance(key, str) or not key for key in values):
            raise TypeError("summary keys must be non-empty strings")
        if self._run is None:
            return
        try:
            with _preserve_global_rngs():
                self._run.summary.update(dict(values))
        except Exception as error:
            _append_failure(self._failure_path, operation="summary", error=error)

    def assert_complete(
        self,
        required_keys: frozenset[str] = REQUIRED_METRIC_KEYS,
    ) -> None:
        """Fail closed unless every metric required by the active mode was observed."""

        if not isinstance(required_keys, frozenset) or any(
            not isinstance(key, str) or not key for key in required_keys
        ):
            raise TypeError("required_keys must be a frozenset of metric names")
        missing = required_keys - self._observed_metric_keys
        if missing:
            raise RuntimeError(f"missing required metrics: {sorted(missing)}")

    def log_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        files: tuple[Path, ...],
        metadata: Mapping[str, object],
    ) -> None:
        """Log one immutable versioned offline artifact from explicit existing files."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact name must be non-empty text")
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise ValueError("artifact_type must be non-empty text")
        if not isinstance(files, tuple) or not files:
            raise ValueError("artifact files must be a non-empty tuple")
        if not isinstance(metadata, Mapping):
            raise TypeError("artifact metadata must be a mapping")
        for path in files:
            if not isinstance(path, Path):
                raise TypeError("artifact files must contain Paths")
            if not path.is_file():
                raise ValueError(f"artifact file does not exist: {path}")
        if self._run is None:
            return
        try:
            with _preserve_global_rngs():
                artifact = wandb.Artifact(
                    name=name.strip(),
                    type=artifact_type.strip(),
                    metadata=dict(metadata),
                )
                for path in files:
                    artifact.add_file(str(path), name=path.name)
                self._run.log_artifact(artifact)
        except Exception as error:
            _append_failure(self._failure_path, operation="artifact", error=error)

    def finish(self, *, exit_code: int) -> None:
        """Finish the local/offline W&B run with its actual process outcome."""

        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("exit_code must be an integer")
        if self._run is None:
            return
        try:
            with _preserve_global_rngs():
                self._run.finish(exit_code=exit_code)
        except Exception as error:
            _append_failure(self._failure_path, operation="finish", error=error)


def create_wandb_logger(  # noqa: PLR0912, PLR0913
    *,
    config: RunConfig,
    state: TrainerState,
    metadata: RunMetadata,
    stamp: str,
    directory: Path | None = None,
    run_factory: Callable[..., object] | None = None,
    additional_summary: Mapping[str, object] | None = None,
) -> WandbLogger:
    """Initialize an offline/online run, falling back locally on every W&B failure."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a TrainerState")
    if not isinstance(metadata, RunMetadata):
        raise TypeError("metadata must be RunMetadata")
    if not isinstance(stamp, str) or not stamp:
        raise ValueError("stamp must be non-empty text")
    if directory is not None and not isinstance(directory, Path):
        raise TypeError("directory must be a Path or None")
    if additional_summary is not None and not isinstance(additional_summary, Mapping):
        raise TypeError("additional_summary must be a mapping or None")
    failure_path = None if directory is None else directory / "wandb_failures.jsonl"
    factory = cast(Callable[..., object], wandb.init) if run_factory is None else run_factory
    environment_mode = os.environ.get("WANDB_MODE")
    effective_mode = config.wandb_mode if environment_mode is None else environment_mode
    if effective_mode not in WANDB_MODES:
        raise ValueError("WANDB_MODE must be disabled, offline, or online")
    arguments: dict[str, object] = {
        "project": PROJECT_NAME,
        "name": f"phase-{config.phase}-{metadata.git_sha[:7]}-seed{config.seed}-{stamp}",
        "tags": (
            f"phase-{config.phase}",
            f"seed-{config.seed}",
            f"arm-{config.arm}",
            f"stage-{config.stage}",
        ),
        "config": asdict(config),
        "mode": effective_mode,
    }
    if state.wandb_run_id:
        arguments["id"] = state.wandb_run_id
        arguments["resume"] = "allow"
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        arguments["dir"] = str(directory)
    if additional_summary is not None and any(
        not isinstance(key, str) or not key for key in additional_summary
    ):
        raise TypeError("additional summary keys must be non-empty strings")

    fallback_payload = json.dumps(
        {
            "config": asdict(config),
            "git_sha": metadata.git_sha,
            "stamp": stamp,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fallback_id = state.wandb_run_id or f"local-{hashlib.sha256(fallback_payload).hexdigest()[:16]}"
    try:
        with _preserve_global_rngs():
            result = factory(**arguments)
        if result is None:
            raise RuntimeError("wandb.init did not return a run")
        run = cast(_Run, result)
        if not isinstance(run.id, str) or not run.id:
            raise RuntimeError("W&B run did not provide a non-empty ID")
    except Exception as error:
        _append_failure(failure_path, operation="init", error=error)
        state.wandb_run_id = fallback_id
        return WandbLogger(
            run=None,
            initial_logging_step=state.logging_step,
            failure_path=failure_path,
        )

    state.wandb_run_id = run.id
    logger = WandbLogger(
        run=run,
        initial_logging_step=state.logging_step,
        failure_path=failure_path,
    )
    logger.update_summary(metadata.as_summary())
    if additional_summary is not None:
        logger.update_summary(additional_summary)
    return logger


def _repository_files(repository: Path) -> tuple[Path, ...]:
    try:
        output = subprocess.run(
            ("git", "-C", str(repository), "ls-files", "-z"),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return tuple(path for path in repository.rglob("*") if path.is_file())
    return tuple(repository / name.decode("utf-8") for name in output.split(b"\0") if name)


def scan_repository_for_credentials(repository: Path) -> tuple[CredentialFinding, ...]:
    """Scan tracked (or fallback local) bytes without returning any credential content."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a Path")
    if not repository.is_dir():
        raise ValueError("repository must be an existing directory")
    findings: list[CredentialFinding] = []
    for path in _repository_files(repository):
        if path.name in CREDENTIAL_FILENAMES:
            findings.append(CredentialFinding(path=path, reason="credential filename"))
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        for reason, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(content):
                findings.append(CredentialFinding(path=path, reason=reason))
                break
    return tuple(findings)
