"""Offline-first W&B initialization, monotonic logging, and credential auditing."""

from __future__ import annotations

import platform
import re
import socket
import subprocess
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch

import wandb
from checkers.config import RunConfig
from checkers.metrics import REQUIRED_METRIC_KEYS
from checkers.trainer_state import TrainerState

PROJECT_NAME = "checkers-ppo"
PROVENANCE_PREFIX = "provenance/"
CREDENTIAL_FILENAMES = frozenset(
    {
        ".env",
        ".secrets",
        ".netrc",
        "kaggle.json",
        "wandb_api_key",
    }
)
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
    """Minimal run wrapper enforcing explicit, monotonic, resumable log steps."""

    def __init__(self, *, run: _Run, initial_logging_step: int) -> None:
        if not hasattr(run, "log") or not hasattr(run, "finish"):
            raise TypeError("run must provide log and finish methods")
        if isinstance(initial_logging_step, bool) or not isinstance(initial_logging_step, int):
            raise TypeError("initial_logging_step must be an integer")
        if initial_logging_step < 0:
            raise ValueError("initial_logging_step must be non-negative")
        self._run = run
        self._last_logging_step = initial_logging_step - 1
        self._observed_metric_keys: set[str] = set()

    @property
    def observed_metric_keys(self) -> frozenset[str]:
        """Return required metric keys observed at least once in this logger session."""

        return frozenset(self._observed_metric_keys)

    def log(self, metrics: Mapping[str, object], *, state: TrainerState) -> None:
        """Log one committed record at the trainer-owned explicit step."""

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
        self._run.log(payload, step=state.logging_step, commit=True)
        self._last_logging_step = state.logging_step
        self._observed_metric_keys.update(REQUIRED_METRIC_KEYS.intersection(payload))
        state.advance_logging_step()

    def assert_complete(self) -> None:
        """Fail closed unless every frozen §13.2 metric has been observed."""

        missing = REQUIRED_METRIC_KEYS - self._observed_metric_keys
        if missing:
            raise RuntimeError(f"missing required metrics: {sorted(missing)}")

    def finish(self, *, exit_code: int) -> None:
        """Finish the local/offline W&B run with its actual process outcome."""

        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("exit_code must be an integer")
        self._run.finish(exit_code=exit_code)


def create_wandb_logger(
    *,
    config: RunConfig,
    state: TrainerState,
    metadata: RunMetadata,
    stamp: str,
    run_factory: Callable[..., object] | None = None,
) -> WandbLogger:
    """Initialize an offline-capable run with exact naming, tags, config, and resume ID."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a TrainerState")
    if not isinstance(metadata, RunMetadata):
        raise TypeError("metadata must be RunMetadata")
    if not isinstance(stamp, str) or not stamp:
        raise ValueError("stamp must be non-empty text")
    factory = cast(Callable[..., object], wandb.init) if run_factory is None else run_factory
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
        "mode": config.wandb_mode,
    }
    if state.wandb_run_id:
        arguments["id"] = state.wandb_run_id
        arguments["resume"] = "allow"
    result = factory(**arguments)
    if result is None:
        raise RuntimeError("wandb.init did not return a run")
    run = cast(_Run, result)
    if not isinstance(run.id, str) or not run.id:
        raise RuntimeError("W&B run did not provide a non-empty ID")
    run.summary.update(metadata.as_summary())
    state.wandb_run_id = run.id
    return WandbLogger(run=run, initial_logging_step=state.logging_step)


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
