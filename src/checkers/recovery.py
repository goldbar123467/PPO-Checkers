"""Fail-closed, non-destructive recovery preparation for interrupted PPO runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import psutil
import torch

from checkers.checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from checkers.config import RunConfig, load_run_config
from checkers.metric_history import SCHEMA as METRIC_HISTORY_SCHEMA
from checkers.rl.networks import CheckersNetwork
from checkers.train import TrainingSession

RECOVERY_SCHEMA = "CHECKERS_PPO_RECOVERY_1"
RECOVERY_STATUS = "PREPARED"
PARTIAL_MARKER_SCHEMA = "CHECKERS_PPO_RECOVERY_PARTIAL_1"
METRIC_FIELDS = frozenset(
    {
        "schema",
        "logging_step",
        "kind",
        "update_idx",
        "global_step",
        "elapsed_training_seconds",
        "metrics",
    }
)
SOFTWARE_PACKAGES = (
    "gymnasium",
    "numpy",
    "psutil",
    "pynvml",
    "PyYAML",
    "torch",
    "wandb",
)


class RecoveryError(RuntimeError):
    """Raised when an interrupted run cannot be recovered without inference."""


@dataclass(frozen=True, slots=True)
class MetricRecordEvidence:
    """Identity and byte location of one validated source metric record."""

    line_number: int
    byte_start: int
    byte_end: int
    sha256: str
    kind: str
    update_idx: int
    global_step: int
    elapsed_training_seconds: float
    logging_step: int

    @property
    def identifier(self) -> str:
        """Return a stable human-readable logical record identifier."""

        return f"{self.kind}:update-{self.update_idx}:logging-step-{self.logging_step}"


@dataclass(frozen=True, slots=True)
class CopiedSource:
    """A source file frozen during recovery analysis."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """Fully analyzed inputs needed for atomic recovery materialization."""

    repository: Path
    source_run_directory: Path
    recovery_run_directory: Path
    source_checkpoint: CopiedSource
    source_checkpoint_sidecar: CopiedSource
    source_metrics: CopiedSource
    source_config: CopiedSource
    source_evaluations: tuple[CopiedSource, ...]
    config: RunConfig
    source_commit: str
    source_git_dirty: bool
    current_commit: str
    working_tree_clean: bool
    checkpoint_update_idx: int
    checkpoint_global_step: int
    checkpoint_elapsed_training_seconds: float
    checkpoint_logging_step: int
    wandb_run_id: str
    records: tuple[MetricRecordEvidence, ...]
    aligned_record_count: int
    aligned_byte_end: int

    @property
    def aligned_records(self) -> tuple[MetricRecordEvidence, ...]:
        """Return the prefix proven consistent with the checkpoint."""

        return self.records[: self.aligned_record_count]

    @property
    def orphaned_records(self) -> tuple[MetricRecordEvidence, ...]:
        """Return valid records written after the durable checkpoint."""

        return self.records[self.aligned_record_count :]

    @property
    def checkpoint_boundary(self) -> MetricRecordEvidence:
        """Return the final record represented by checkpoint logging state."""

        return self.aligned_records[-1]


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Paths and status produced by a completed preparation operation."""

    recovery_run_directory: Path
    manifest_path: Path
    recovered_checkpoint_path: Path
    metrics_path: Path
    orphaned_metrics_path: Path
    status: str


@dataclass(frozen=True, slots=True)
class RecoveryResumeContext:
    """Validated local recovery provenance supplied to training and W&B."""

    manifest_path: Path
    manifest_sha256: str
    source_commit: str
    recovery_commit: str
    checkpoint_update_idx: int
    source_checkpoint_sha256: str
    source_checkpoint_name: str
    source_wandb_run_id: str
    artifact_files: tuple[Path, ...]

    def wandb_summary(self) -> dict[str, object]:
        """Return non-secret recovery provenance without local absolute paths."""

        return {
            "recovery/is_recovery": True,
            "recovery/manifest_sha256": self.manifest_sha256,
            "recovery/source_commit": self.source_commit,
            "recovery/current_commit": self.recovery_commit,
            "recovery/checkpoint_update": self.checkpoint_update_idx,
            "recovery/checkpoint_sha256": self.source_checkpoint_sha256,
            "recovery/checkpoint_name": self.source_checkpoint_name,
            "recovery/source_wandb_run_id": self.source_wandb_run_id,
        }


@dataclass(frozen=True, slots=True)
class SmokeAuditResult:
    """Machine-readable evidence from one bounded recovery continuation."""

    status: str
    audit_path: Path
    report_path: Path
    checkpoint_path: Path
    end_update: int
    logging_step: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file without following mutable state twice."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.is_file():
        raise RecoveryError(f"required recovery source is not a file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(path: Path) -> CopiedSource:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RecoveryError(f"recovery source must be a regular file: {resolved}")
    return CopiedSource(
        path=resolved,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryError(f"metric {name} must be an integer")
    if value < 0:
        raise RecoveryError(f"metric {name} must be non-negative")
    return value


def _strict_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"metric {name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise RecoveryError(f"metric {name} must be finite and non-negative")
    return checked


def _parse_metric_records(  # noqa: PLR0912, PLR0915
    payload: bytes, *, batch_size: int
) -> tuple[MetricRecordEvidence, ...]:
    if not payload:
        raise RecoveryError("source metric history is empty")
    lines = payload.splitlines(keepends=True)
    records: list[MetricRecordEvidence] = []
    byte_start = 0
    logical_identifiers: set[tuple[str, int]] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        byte_end = byte_start + len(raw_line)
        encoded = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
        if encoded.endswith(b"\r"):
            encoded = encoded[:-1]
        if not encoded:
            raise RecoveryError(
                f"metric history contains a blank or partial record at line {line_number}, "
                f"byte {byte_start}"
            )
        try:
            raw_value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecoveryError(
                f"metric history contains malformed JSON at line {line_number}, "
                f"byte {byte_start}: {error}"
            ) from error
        if not isinstance(raw_value, dict) or set(raw_value) != METRIC_FIELDS:
            raise RecoveryError(f"metric record fields are invalid at line {line_number}")
        value = cast(dict[str, object], raw_value)
        if value["schema"] != METRIC_HISTORY_SCHEMA:
            raise RecoveryError(f"metric schema is invalid at line {line_number}")
        kind = value["kind"]
        if not isinstance(kind, str) or not kind:
            raise RecoveryError(f"metric kind is invalid at line {line_number}")
        logging_step = _strict_int(value["logging_step"], "logging_step")
        update_idx = _strict_int(value["update_idx"], "update_idx")
        global_step = _strict_int(value["global_step"], "global_step")
        elapsed = _strict_float(value["elapsed_training_seconds"], "elapsed_training_seconds")
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise RecoveryError(f"metric payload is invalid at line {line_number}")
        for name, metric_value in metrics.items():
            if not isinstance(name, str) or not name:
                raise RecoveryError(f"metric name is invalid at line {line_number}")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise RecoveryError(f"metric value {name!r} is not numeric at line {line_number}")
            if not math.isfinite(float(metric_value)):
                raise RecoveryError(f"metric value {name!r} is not finite at line {line_number}")
        if logging_step != line_number - 1:
            raise RecoveryError(
                f"metric logging steps are not contiguous at line {line_number}: "
                f"expected {line_number - 1}, observed {logging_step}"
            )
        if global_step != update_idx * batch_size:
            raise RecoveryError(f"metric transition count is inconsistent at line {line_number}")
        logical_identifier = (kind, update_idx)
        if logical_identifier in logical_identifiers:
            raise RecoveryError(
                f"duplicate logical metric record {kind}/update-{update_idx} at line {line_number}"
            )
        logical_identifiers.add(logical_identifier)
        if records:
            previous = records[-1]
            if update_idx < previous.update_idx or global_step < previous.global_step:
                raise RecoveryError(f"metric progress regresses at line {line_number}")
            if elapsed < previous.elapsed_training_seconds:
                raise RecoveryError(f"metric elapsed time regresses at line {line_number}")
        records.append(
            MetricRecordEvidence(
                line_number=line_number,
                byte_start=byte_start,
                byte_end=byte_end,
                sha256=_sha256_bytes(raw_line),
                kind=kind,
                update_idx=update_idx,
                global_step=global_step,
                elapsed_training_seconds=elapsed,
                logging_step=logging_step,
            )
        )
        byte_start = byte_end
    return tuple(records)


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RecoveryError(f"Git provenance query failed: {' '.join(arguments)}") from error
    return result.stdout.strip()


def _load_validated_checkpoint(
    checkpoint: Path, config: RunConfig
) -> tuple[LoadedCheckpoint, torch.optim.Optimizer]:
    network = CheckersNetwork()
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        eps=config.adam_eps,
    )
    try:
        loaded = load_checkpoint(
            path=checkpoint,
            expected_config=config,
            network=network,
            optimizer=optimizer,
        )
    except CheckpointError as error:
        raise RecoveryError(f"checkpoint validation failed: {error}") from error
    return loaded, optimizer


def _validate_alignment(
    records: tuple[MetricRecordEvidence, ...], loaded: LoadedCheckpoint
) -> tuple[int, int]:
    state = loaded.state
    if state.logging_step < 1:
        raise RecoveryError("checkpoint has no preceding metric record")
    candidates = tuple(
        record
        for record in records
        if record.logging_step == state.logging_step - 1
        and record.update_idx == state.update_idx
        and record.global_step == state.global_step
        and math.isclose(
            record.elapsed_training_seconds,
            state.elapsed_training_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if len(candidates) != 1:
        raise RecoveryError(
            "metric history does not contain one unique checkpoint-aligned boundary "
            f"for logging step {state.logging_step - 1}"
        )
    boundary = candidates[0]
    aligned_count = boundary.line_number
    if aligned_count != state.logging_step:
        raise RecoveryError("checkpoint logging state does not equal the aligned record count")
    for orphan in records[aligned_count:]:
        if orphan.update_idx <= state.update_idx:
            raise RecoveryError(
                "post-boundary metrics do not prove a strictly later trainer update"
            )
    return aligned_count, boundary.byte_end


def analyze_recovery(
    *,
    repository: Path,
    source_run_directory: Path,
    recovery_run_directory: Path,
    checkpoint_path: Path,
) -> RecoveryPlan:
    """Analyze immutable recovery inputs without creating any destination files."""

    if not all(
        isinstance(path, Path)
        for path in (
            repository,
            source_run_directory,
            recovery_run_directory,
            checkpoint_path,
        )
    ):
        raise TypeError("recovery paths must be Paths")
    checked_repository = repository.resolve(strict=True)
    checked_source_run = source_run_directory.resolve(strict=True)
    if not checked_repository.is_dir() or not checked_source_run.is_dir():
        raise RecoveryError("repository and source run must be existing directories")
    checked_destination = recovery_run_directory.resolve(strict=False)
    if checked_destination == checked_source_run:
        raise RecoveryError("recovery destination must differ from the source run")
    if (
        checked_source_run in checked_destination.parents
        or checked_destination in checked_source_run.parents
    ):
        raise RecoveryError(
            "recovery destination must be a sibling, not nested with the source run"
        )
    checked_checkpoint = checkpoint_path.resolve(strict=True)
    if checked_source_run not in checked_checkpoint.parents:
        raise RecoveryError("source checkpoint must reside beneath the source run")
    source_metrics = _source(checked_source_run / "metrics.jsonl")
    source_config = _source(checked_source_run / "config.resolved.yaml")
    source_checkpoint = _source(checked_checkpoint)
    source_sidecar = _source(checked_checkpoint.with_suffix(f"{checked_checkpoint.suffix}.sha256"))
    config = load_run_config(source_config.path.read_text(encoding="utf-8"))
    loaded, _optimizer = _load_validated_checkpoint(source_checkpoint.path, config)
    records = _parse_metric_records(
        source_metrics.path.read_bytes(),
        batch_size=config.batch_size,
    )
    aligned_count, aligned_byte_end = _validate_alignment(records, loaded)
    evaluations: list[CopiedSource] = []
    evaluation_directory = checked_source_run / "evaluations"
    if evaluation_directory.exists():
        for path in sorted(evaluation_directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RecoveryError(f"source evaluation is malformed: {path}") from error
            if not isinstance(value, dict):
                raise RecoveryError(f"source evaluation root is invalid: {path}")
            update_idx = _strict_int(value.get("update_idx"), "evaluation update_idx")
            if update_idx <= loaded.state.update_idx:
                evaluations.append(_source(path))
    current_commit = _git_output(checked_repository, "rev-parse", "HEAD")
    working_tree_clean = not bool(
        _git_output(checked_repository, "status", "--porcelain", "--untracked-files=normal")
    )
    return RecoveryPlan(
        repository=checked_repository,
        source_run_directory=checked_source_run,
        recovery_run_directory=checked_destination,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sidecar=source_sidecar,
        source_metrics=source_metrics,
        source_config=source_config,
        source_evaluations=tuple(evaluations),
        config=config,
        source_commit=loaded.git_sha,
        source_git_dirty=loaded.git_dirty,
        current_commit=current_commit,
        working_tree_clean=working_tree_clean,
        checkpoint_update_idx=loaded.state.update_idx,
        checkpoint_global_step=loaded.state.global_step,
        checkpoint_elapsed_training_seconds=loaded.state.elapsed_training_seconds,
        checkpoint_logging_step=loaded.state.logging_step,
        wandb_run_id=loaded.state.wandb_run_id,
        records=records,
        aligned_record_count=aligned_count,
        aligned_byte_end=aligned_byte_end,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file(source: CopiedSource, destination: Path) -> None:
    if sha256_file(source.path) != source.sha256 or source.path.stat().st_size != source.size_bytes:
        raise RecoveryError(f"source changed after inspection: {source.path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.path.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if sha256_file(destination) != source.sha256:
        raise RecoveryError(f"copied recovery file failed hash verification: {destination}")
    if sha256_file(source.path) != source.sha256:
        raise RecoveryError(f"source changed during recovery copy: {source.path}")


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in SOFTWARE_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _cuda_information() -> dict[str, object]:
    information: dict[str, object] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        information.update(
            {
                "current_device": index,
                "device_name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return information


def _host_information() -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "ram_total_bytes": memory.total,
    }


def _file_entry(
    *,
    root: Path,
    path: Path,
    operation: str,
    source: Path | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": str(path.relative_to(root)),
        "operation": operation,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if source is not None:
        entry["source"] = str(source)
    return entry


def _recovery_report(plan: RecoveryPlan, *, partial_output_detected: bool) -> str:
    boundary = plan.checkpoint_boundary
    orphan_lines = ", ".join(str(record.line_number) for record in plan.orphaned_records) or "none"
    return (
        "# Phase 7 recovery preparation report\n\n"
        f"Status: `{RECOVERY_STATUS}`\n\n"
        f"Source run: `{plan.source_run_directory}`\n\n"
        f"Recovery run: `{plan.recovery_run_directory}`\n\n"
        f"Source checkpoint: `{plan.source_checkpoint.path}`\n\n"
        f"Checkpoint update/global step: {plan.checkpoint_update_idx} / "
        f"{plan.checkpoint_global_step}\n\n"
        f"Checkpoint elapsed training seconds: "
        f"{plan.checkpoint_elapsed_training_seconds:.9f}\n\n"
        f"Checkpoint-aligned boundary: source line {boundary.line_number}, logging step "
        f"{boundary.logging_step}, byte end {boundary.byte_end}.\n\n"
        f"Orphaned source metric lines: {orphan_lines}. They are preserved verbatim and excluded "
        "because no matching model, optimizer, collector, trainer, and RNG state exists.\n\n"
        f"Interrupted partial destination detected before preparation: "
        f"{str(partial_output_detected).lower()}.\n\n"
        "The original run was read only. This directory is prepared for a checkpoint-aligned "
        "resume; it is not evidence that the resume or final experiment completed.\n"
    )


def _verify_existing_result(plan: RecoveryPlan) -> RecoveryResult:
    manifest_path = plan.recovery_run_directory / "recovery" / "recovery-manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(
            "existing recovery destination is not a valid prepared recovery"
        ) from error
    if not isinstance(value, dict):
        raise RecoveryError("existing recovery manifest root is invalid")
    if value.get("schema") != RECOVERY_SCHEMA or value.get("status") != RECOVERY_STATUS:
        raise RecoveryError("existing recovery destination has incompatible status")
    if value.get("source_metrics_sha256") != plan.source_metrics.sha256:
        raise RecoveryError("existing recovery destination refers to different source metrics")
    if value.get("source_checkpoint_sha256") != plan.source_checkpoint.sha256:
        raise RecoveryError("existing recovery destination refers to a different checkpoint")
    checkpoint = plan.recovery_run_directory / "checkpoints" / plan.source_checkpoint.path.name
    metrics = plan.recovery_run_directory / "metrics.jsonl"
    orphaned = plan.recovery_run_directory / "recovery" / "orphaned-metrics.jsonl"
    expected_active = value.get("active_metrics_sha256")
    expected_orphaned = value.get("orphaned_metrics_sha256")
    if (
        not isinstance(expected_active, str)
        or not isinstance(expected_orphaned, str)
        or sha256_file(checkpoint) != plan.source_checkpoint.sha256
        or sha256_file(metrics) != expected_active
        or sha256_file(orphaned) != expected_orphaned
    ):
        raise RecoveryError("existing recovery destination failed its immutable preparation audit")
    return RecoveryResult(
        recovery_run_directory=plan.recovery_run_directory,
        manifest_path=manifest_path,
        recovered_checkpoint_path=checkpoint,
        metrics_path=metrics,
        orphaned_metrics_path=orphaned,
        status=RECOVERY_STATUS,
    )


def _required_manifest_text(manifest: Mapping[str, object], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"recovery manifest field {name} must be non-empty text")
    return value


def _required_manifest_int(manifest: Mapping[str, object], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecoveryError(f"recovery manifest field {name} must be a non-negative integer")
    return value


def validate_recovery_resume_context(  # noqa: PLR0912, PLR0915
    *,
    output_directory: Path,
    resume_path: Path | None,
    current_commit: str,
    working_tree_clean: bool,
) -> RecoveryResumeContext | None:
    """Validate prepared recovery provenance before training creates a logger."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a Path")
    if resume_path is not None and not isinstance(resume_path, Path):
        raise TypeError("resume_path must be a Path or None")
    if not isinstance(current_commit, str) or not current_commit:
        raise ValueError("current_commit must be non-empty text")
    if not isinstance(working_tree_clean, bool):
        raise TypeError("working_tree_clean must be bool")
    manifest_path = output_directory / "recovery" / "recovery-manifest.json"
    if not manifest_path.exists():
        return None
    if resume_path is None:
        raise RecoveryError("a prepared recovery directory requires an explicit checkpoint resume")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError("recovery manifest is unreadable") from error
    if not isinstance(raw_manifest, dict):
        raise RecoveryError("recovery manifest root must be a mapping")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("schema") != RECOVERY_SCHEMA or manifest.get("status") != RECOVERY_STATUS:
        raise RecoveryError("recovery manifest schema or status is invalid")
    expected_output = Path(_required_manifest_text(manifest, "recovery_run_directory")).resolve(
        strict=False
    )
    if output_directory.resolve(strict=True) != expected_output:
        raise RecoveryError("training output directory disagrees with the recovery manifest")
    expected_commit = _required_manifest_text(manifest, "current_commit")
    if current_commit != expected_commit:
        raise RecoveryError("current source commit differs from the prepared recovery commit")
    prepared_clean = manifest.get("working_tree_clean")
    if not isinstance(prepared_clean, bool) or working_tree_clean != prepared_clean:
        raise RecoveryError("working-tree cleanliness differs from recovery preparation")
    checked_resume = resume_path.resolve(strict=True)
    checkpoint_directory = (output_directory / "checkpoints").resolve(strict=True)
    if checked_resume.parent != checkpoint_directory:
        raise RecoveryError(
            "resume checkpoint must be inside the recovery run checkpoint directory"
        )
    recovered_checkpoint = Path(
        _required_manifest_text(manifest, "recovered_checkpoint_path")
    ).resolve(strict=True)
    source_checkpoint = Path(_required_manifest_text(manifest, "source_checkpoint_path")).resolve(
        strict=True
    )
    source_metrics = Path(_required_manifest_text(manifest, "source_metrics_path")).resolve(
        strict=True
    )
    checkpoint_sha256 = _required_manifest_text(manifest, "source_checkpoint_sha256")
    source_metrics_sha256 = _required_manifest_text(manifest, "source_metrics_sha256")
    if sha256_file(source_checkpoint) != checkpoint_sha256:
        raise RecoveryError("original source checkpoint changed after recovery preparation")
    if sha256_file(source_metrics) != source_metrics_sha256:
        raise RecoveryError("original source metrics changed after recovery preparation")
    if sha256_file(recovered_checkpoint) != checkpoint_sha256:
        raise RecoveryError("recovered source checkpoint copy failed provenance validation")
    if checked_resume == recovered_checkpoint:
        active_metrics_sha256 = _required_manifest_text(manifest, "active_metrics_sha256")
        if sha256_file(output_directory / "metrics.jsonl") != active_metrics_sha256:
            raise RecoveryError("checkpoint-aligned active metrics changed before initial resume")
    artifact_names = (
        "recovery-manifest.json",
        "recovery-report.md",
        "orphaned-metrics.jsonl",
        "source-checkpoint-sha256.txt",
        "source-metrics-sha256.txt",
    )
    artifact_files = tuple(output_directory / "recovery" / name for name in artifact_names)
    if not all(path.is_file() for path in artifact_files):
        raise RecoveryError("recovery artifact set is incomplete")
    return RecoveryResumeContext(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        source_commit=_required_manifest_text(manifest, "source_commit"),
        recovery_commit=expected_commit,
        checkpoint_update_idx=_required_manifest_int(manifest, "checkpoint_update_number"),
        source_checkpoint_sha256=checkpoint_sha256,
        source_checkpoint_name=recovered_checkpoint.name,
        source_wandb_run_id=_required_manifest_text(manifest, "wandb_run_id"),
        artifact_files=artifact_files,
    )


def materialize_recovery(  # noqa: PLR0912, PLR0915
    plan: RecoveryPlan, *, recovery_command: str
) -> RecoveryResult:
    """Create a verified recovery directory atomically from an analyzed plan."""

    if not isinstance(plan, RecoveryPlan):
        raise TypeError("plan must be a RecoveryPlan")
    if not isinstance(recovery_command, str) or not recovery_command.strip():
        raise ValueError("recovery_command must be non-empty text")
    destination = plan.recovery_run_directory
    if destination.exists():
        return _verify_existing_result(plan)
    stage = destination.with_name(f".{destination.name}.partial")
    partial_output_detected = stage.exists()
    if partial_output_detected:
        if stage.parent != destination.parent or stage.name != f".{destination.name}.partial":
            raise RecoveryError("refusing to clean an unexpected partial recovery path")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    marker = {
        "schema": PARTIAL_MARKER_SCHEMA,
        "source_run_directory": str(plan.source_run_directory),
        "recovery_run_directory": str(destination),
        "source_checkpoint_sha256": plan.source_checkpoint.sha256,
        "source_metrics_sha256": plan.source_metrics.sha256,
    }
    _atomic_write(
        stage / ".recovery-partial.json",
        (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    try:
        source_metrics_payload = plan.source_metrics.path.read_bytes()
        if (
            _sha256_bytes(source_metrics_payload) != plan.source_metrics.sha256
            or len(source_metrics_payload) != plan.source_metrics.size_bytes
        ):
            raise RecoveryError("source metrics changed after recovery analysis")
        records = _parse_metric_records(source_metrics_payload, batch_size=plan.config.batch_size)
        if records != plan.records:
            raise RecoveryError("source metric identities changed after recovery analysis")
        active_payload = source_metrics_payload[: plan.aligned_byte_end]
        orphaned_payload = source_metrics_payload[plan.aligned_byte_end :]
        metrics_path = stage / "metrics.jsonl"
        orphaned_path = stage / "recovery" / "orphaned-metrics.jsonl"
        _atomic_write(metrics_path, active_payload)
        _atomic_write(orphaned_path, orphaned_payload)
        checkpoint_path = stage / "checkpoints" / plan.source_checkpoint.path.name
        _copy_file(plan.source_checkpoint, checkpoint_path)
        _copy_file(
            plan.source_checkpoint_sidecar,
            checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.sha256"),
        )
        top_level_config = stage / "config.resolved.yaml"
        nested_config = stage / "config" / "config.resolved.yaml"
        _copy_file(plan.source_config, top_level_config)
        _copy_file(plan.source_config, nested_config)
        evaluation_entries: list[dict[str, object]] = []
        for evaluation in plan.source_evaluations:
            evaluation_path = stage / "evaluations" / evaluation.path.name
            _copy_file(evaluation, evaluation_path)
            evaluation_entries.append(
                _file_entry(
                    root=stage,
                    path=evaluation_path,
                    operation="copied-checkpoint-aligned-evaluation",
                    source=evaluation.path,
                )
            )
        (stage / "logs").mkdir()
        checkpoint_hash_path = stage / "recovery" / "source-checkpoint-sha256.txt"
        metrics_hash_path = stage / "recovery" / "source-metrics-sha256.txt"
        _atomic_write(
            checkpoint_hash_path,
            f"{plan.source_checkpoint.sha256}  {plan.source_checkpoint.path.name}\n".encode(
                "ascii"
            ),
        )
        _atomic_write(
            metrics_hash_path,
            f"{plan.source_metrics.sha256}  {plan.source_metrics.path.name}\n".encode("ascii"),
        )
        report_path = stage / "recovery" / "recovery-report.md"
        _atomic_write(
            report_path,
            _recovery_report(
                plan,
                partial_output_detected=partial_output_detected,
            ).encode("utf-8"),
        )
        files: list[dict[str, object]] = [
            _file_entry(
                root=stage,
                path=metrics_path,
                operation="transformed-checkpoint-aligned-prefix",
                source=plan.source_metrics.path,
            ),
            _file_entry(
                root=stage,
                path=orphaned_path,
                operation="transformed-verbatim-orphaned-suffix",
                source=plan.source_metrics.path,
            ),
            _file_entry(
                root=stage,
                path=checkpoint_path,
                operation="copied-verified-checkpoint",
                source=plan.source_checkpoint.path,
            ),
            _file_entry(
                root=stage,
                path=checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.sha256"),
                operation="copied-checkpoint-sidecar",
                source=plan.source_checkpoint_sidecar.path,
            ),
            _file_entry(
                root=stage,
                path=top_level_config,
                operation="copied-resume-config",
                source=plan.source_config.path,
            ),
            _file_entry(
                root=stage,
                path=nested_config,
                operation="copied-audit-config",
                source=plan.source_config.path,
            ),
            _file_entry(
                root=stage,
                path=checkpoint_hash_path,
                operation="generated",
            ),
            _file_entry(
                root=stage,
                path=metrics_hash_path,
                operation="generated",
            ),
            _file_entry(root=stage, path=report_path, operation="generated"),
            *evaluation_entries,
            {
                "path": "recovery/recovery-manifest.json",
                "operation": "generated-self-describing-manifest",
                "size_bytes": None,
                "sha256": None,
            },
        ]
        boundary = plan.checkpoint_boundary
        orphaned_records = [
            {
                **asdict(record),
                "identifier": record.identifier,
                "exclusion_reason": (
                    "No matching durable model, optimizer, collector, trainer, and RNG state exists"
                ),
            }
            for record in plan.orphaned_records
        ]
        timestamp = datetime.now(UTC).isoformat()
        manifest: dict[str, object] = {
            "schema": RECOVERY_SCHEMA,
            "status": RECOVERY_STATUS,
            "source_commit": plan.source_commit,
            "source_git_dirty": plan.source_git_dirty,
            "current_commit": plan.current_commit,
            "working_tree_clean": plan.working_tree_clean,
            "source_run_directory": str(plan.source_run_directory),
            "recovery_run_directory": str(destination),
            "source_checkpoint_path": str(plan.source_checkpoint.path),
            "recovered_checkpoint_path": str(
                destination / "checkpoints" / plan.source_checkpoint.path.name
            ),
            "source_checkpoint_sha256": plan.source_checkpoint.sha256,
            "source_metrics_path": str(plan.source_metrics.path),
            "source_metrics_sha256": plan.source_metrics.sha256,
            "active_metrics_sha256": _sha256_bytes(active_payload),
            "orphaned_metrics_sha256": _sha256_bytes(orphaned_payload),
            "checkpoint_update_number": plan.checkpoint_update_idx,
            "checkpoint_transition_count": plan.checkpoint_global_step,
            "checkpoint_elapsed_training_seconds": plan.checkpoint_elapsed_training_seconds,
            "expected_logging_step": plan.checkpoint_logging_step,
            "observed_logging_step": plan.records[-1].logging_step,
            "orphaned_metric_record_count": len(plan.orphaned_records),
            "orphaned_record_identifiers": [record.identifier for record in plan.orphaned_records],
            "orphaned_records": orphaned_records,
            "checkpoint_aligned_record_boundary": {
                **asdict(boundary),
                "identifier": boundary.identifier,
                "aligned_record_count": plan.aligned_record_count,
            },
            "recovery_command": recovery_command.strip(),
            "recovery_timestamp": timestamp,
            "host_information": _host_information(),
            "cuda_device_information": _cuda_information(),
            "software_dependency_versions": _package_versions(),
            "wandb_run_id": plan.wandb_run_id,
            "partial_output_detected": partial_output_detected,
            "original_run_mutated": False,
            "files": sorted(files, key=lambda entry: cast(str, entry["path"])),
            "final_recovery_status": RECOVERY_STATUS,
        }
        manifest_path = stage / "recovery" / "recovery-manifest.json"
        _atomic_write(
            manifest_path,
            (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        if sha256_file(plan.source_metrics.path) != plan.source_metrics.sha256:
            raise RecoveryError("source metrics changed while recovery was materialized")
        if sha256_file(plan.source_checkpoint.path) != plan.source_checkpoint.sha256:
            raise RecoveryError("source checkpoint changed while recovery was materialized")
        (stage / ".recovery-partial.json").unlink()
        directory_descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        stage.replace(destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        raise
    return RecoveryResult(
        recovery_run_directory=destination,
        manifest_path=destination / "recovery" / "recovery-manifest.json",
        recovered_checkpoint_path=destination / "checkpoints" / plan.source_checkpoint.path.name,
        metrics_path=destination / "metrics.jsonl",
        orphaned_metrics_path=destination / "recovery" / "orphaned-metrics.jsonl",
        status=RECOVERY_STATUS,
    )


def prepare_recovery(
    *,
    repository: Path,
    source_run_directory: Path,
    recovery_run_directory: Path,
    checkpoint_path: Path,
    recovery_command: str | None = None,
) -> RecoveryResult:
    """Analyze and atomically prepare a non-destructive recovery directory."""

    plan = analyze_recovery(
        repository=repository,
        source_run_directory=source_run_directory,
        recovery_run_directory=recovery_run_directory,
        checkpoint_path=checkpoint_path,
    )
    command = " ".join(sys.argv) if recovery_command is None else recovery_command
    return materialize_recovery(plan, recovery_command=command)


def _validate_optimizer_devices(
    optimizer: torch.optim.Optimizer,
    *,
    expected_device_type: str,
) -> dict[str, object]:
    """Validate Adam parameter/moment placement while allowing host scalar steps."""

    if not optimizer.state:
        raise RecoveryError("smoke optimizer state is empty")
    parameter_count = 0
    moment_tensor_count = 0
    scalar_step_count = 0
    scalar_step_devices: set[str] = set()
    for parameter, raw_state in optimizer.state.items():
        if parameter.device.type != expected_device_type:
            raise RecoveryError("smoke optimizer parameter is not on the configured device")
        parameter_count += 1
        if not isinstance(raw_state, Mapping):
            raise RecoveryError("smoke optimizer parameter state is not a mapping")
        for name, value in raw_state.items():
            if not isinstance(value, torch.Tensor):
                raise RecoveryError("smoke optimizer state contains a non-tensor value")
            if not bool(torch.isfinite(value).all().item()):
                raise RecoveryError("smoke optimizer state contains a non-finite tensor")
            if name == "step":
                if value.numel() != 1:
                    raise RecoveryError("smoke optimizer step state is not scalar")
                scalar_step_count += 1
                scalar_step_devices.add(str(value.device))
            else:
                if value.device.type != expected_device_type:
                    raise RecoveryError(
                        "smoke optimizer moment state is not on the configured device"
                    )
                moment_tensor_count += 1
    if parameter_count < 1 or moment_tensor_count < 1 or scalar_step_count < 1:
        raise RecoveryError("smoke optimizer state is incomplete")
    return {
        "parameter_and_moment_device": expected_device_type,
        "parameter_count": parameter_count,
        "moment_tensor_count": moment_tensor_count,
        "scalar_step_count": scalar_step_count,
        "scalar_step_devices": sorted(scalar_step_devices),
        "scalar_step_policy": (
            "finite scalar Adam steps may remain on CPU when capturable/fused mode is disabled"
        ),
    }


def _latest_checkpoint_path(run_directory: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in (run_directory / "checkpoints").glob("update-*.pt"):
        stem = path.stem.removeprefix("update-")
        sidecar = path.with_suffix(f"{path.suffix}.sha256")
        if stem.isdigit() and sidecar.is_file():
            candidates.append((int(stem), path))
    if not candidates:
        raise RecoveryError("recovery smoke produced no durable checkpoint")
    return max(candidates, key=lambda item: item[0])[1]


def _load_recovery_manifest(run_directory: Path) -> tuple[Path, dict[str, object]]:
    path = run_directory / "recovery" / "recovery-manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError("recovery smoke manifest is unreadable") from error
    if not isinstance(value, dict):
        raise RecoveryError("recovery smoke manifest root is invalid")
    manifest = cast(dict[str, object], value)
    if manifest.get("schema") != RECOVERY_SCHEMA or manifest.get("status") != RECOVERY_STATUS:
        raise RecoveryError("recovery smoke manifest schema or status is invalid")
    return path, manifest


def _latest_training_manifest(run_directory: Path) -> dict[str, object]:
    candidates: list[tuple[int, dict[str, object]]] = []
    for path in run_directory.glob("manifest-*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        end_update = value.get("end_update")
        if isinstance(end_update, int) and not isinstance(end_update, bool):
            candidates.append((end_update, cast(dict[str, object], value)))
    if not candidates:
        raise RecoveryError("recovery smoke produced no training invocation manifest")
    manifest = max(candidates, key=lambda item: item[0])[1]
    if manifest.get("status") != "completed":
        raise RecoveryError("recovery smoke invocation did not complete")
    return manifest


def audit_recovery_smoke(  # noqa: PLR0912, PLR0915
    *,
    run_directory: Path,
    expected_updates: int = 1,
) -> SmokeAuditResult:
    """Verify a bounded continuation without treating it as the final baseline."""

    if not isinstance(run_directory, Path):
        raise TypeError("run_directory must be a Path")
    if (
        isinstance(expected_updates, bool)
        or not isinstance(expected_updates, int)
        or expected_updates < 1
    ):
        raise ValueError("expected_updates must be a positive integer")
    checked_run = run_directory.resolve(strict=True)
    if not checked_run.is_dir():
        raise RecoveryError("recovery smoke run directory does not exist")
    manifest_path, manifest = _load_recovery_manifest(checked_run)
    config = load_run_config((checked_run / "config.resolved.yaml").read_text(encoding="utf-8"))
    source_update = _required_manifest_int(manifest, "checkpoint_update_number")
    source_global_step = _required_manifest_int(manifest, "checkpoint_transition_count")
    source_logging_step = _required_manifest_int(manifest, "expected_logging_step")
    source_elapsed = manifest.get("checkpoint_elapsed_training_seconds")
    if isinstance(source_elapsed, bool) or not isinstance(source_elapsed, (int, float)):
        raise RecoveryError("recovery manifest checkpoint elapsed time is invalid")
    source_elapsed_float = float(source_elapsed)
    source_checkpoint = Path(_required_manifest_text(manifest, "source_checkpoint_path")).resolve(
        strict=True
    )
    source_metrics = Path(_required_manifest_text(manifest, "source_metrics_path")).resolve(
        strict=True
    )
    if sha256_file(source_checkpoint) != _required_manifest_text(
        manifest, "source_checkpoint_sha256"
    ):
        raise RecoveryError("source checkpoint changed before smoke audit")
    if sha256_file(source_metrics) != _required_manifest_text(manifest, "source_metrics_sha256"):
        raise RecoveryError("source metrics changed before smoke audit")
    active_metrics = checked_run / "metrics.jsonl"
    metrics_payload = active_metrics.read_bytes()
    boundary = manifest.get("checkpoint_aligned_record_boundary")
    if not isinstance(boundary, dict):
        raise RecoveryError("recovery manifest aligned boundary is invalid")
    byte_end = _required_manifest_int(cast(dict[str, object], boundary), "byte_end")
    if _sha256_bytes(metrics_payload[:byte_end]) != _required_manifest_text(
        manifest, "active_metrics_sha256"
    ):
        raise RecoveryError("smoke continuation overwrote the checkpoint-aligned metric prefix")
    records = _parse_metric_records(metrics_payload, batch_size=config.batch_size)
    new_records = records[source_logging_step:]
    new_training_records = tuple(record for record in new_records if record.kind == "training")
    if len(new_training_records) != expected_updates:
        raise RecoveryError("smoke continuation emitted an unexpected number of training records")
    if [record.update_idx for record in new_training_records] != list(
        range(source_update + 1, source_update + expected_updates + 1)
    ):
        raise RecoveryError("smoke continuation update sequence is not contiguous")
    if not new_records or new_records[0].logging_step != source_logging_step:
        raise RecoveryError("smoke continuation began at the wrong logging step")
    latest_checkpoint = _latest_checkpoint_path(checked_run)
    loaded, optimizer = _load_validated_checkpoint(latest_checkpoint, config)
    expected_end_update = source_update + expected_updates
    if loaded.state.update_idx != expected_end_update:
        raise RecoveryError("smoke checkpoint ended at the wrong update")
    if loaded.state.global_step != source_global_step + expected_updates * config.batch_size:
        raise RecoveryError("smoke checkpoint transition count is not monotonic")
    if loaded.state.elapsed_training_seconds <= source_elapsed_float:
        raise RecoveryError("smoke checkpoint elapsed training time did not advance")
    if loaded.state.logging_step != len(records):
        raise RecoveryError("smoke checkpoint and metric logging steps disagree")
    if loaded.state.wandb_run_id != _required_manifest_text(manifest, "wandb_run_id"):
        raise RecoveryError("smoke checkpoint changed the stable W&B run identity")
    expected_device_type = torch.device(config.device).type
    optimizer_device_evidence = _validate_optimizer_devices(
        optimizer,
        expected_device_type=expected_device_type,
    )
    restored = TrainingSession.resume(config=config, checkpoint_path=latest_checkpoint)
    if restored.state.rng_states is None or restored.state.update_idx != expected_end_update:
        raise RecoveryError("smoke checkpoint did not restore trainer and RNG state")
    latest_training = new_training_records[-1]
    raw_record = json.loads(
        metrics_payload[latest_training.byte_start : latest_training.byte_end].decode("utf-8")
    )
    if not isinstance(raw_record, dict) or not isinstance(raw_record.get("metrics"), dict):
        raise RecoveryError("smoke training metric payload is invalid")
    metrics = cast(dict[str, object], raw_record["metrics"])
    for key in ("mask/sample_legality_violations", "mask/oracle_disagreements"):
        if metrics.get(key) != 0.0:
            raise RecoveryError(f"smoke correctness counter is nonzero: {key}")
    required_system_metrics = {
        "system/cpu_percent",
        "system/ram_used_bytes",
        "system/ram_available_bytes",
        "system/process_cpu_percent",
        "system/process_rss_bytes",
    }
    if config.device == "cuda":
        required_system_metrics.update(
            {
                "system/gpu_utilization_percent",
                "system/gpu_memory_used_mib",
                "system/gpu_temperature_celsius",
                "system/gpu_power_draw_watts",
            }
        )
    if not required_system_metrics.issubset(metrics):
        raise RecoveryError("smoke metrics omit required host or device telemetry")
    training_manifest = _latest_training_manifest(checked_run)
    if training_manifest.get("start_update") != source_update:
        raise RecoveryError("smoke invocation manifest has the wrong start update")
    if training_manifest.get("end_update") != expected_end_update:
        raise RecoveryError("smoke invocation manifest has the wrong end update")
    evidence = {
        "schema": "CHECKERS_PPO_RECOVERY_SMOKE_AUDIT_1",
        "status": "PASS",
        "classification": "BOUNDED_RECOVERY_SMOKE_NOT_FINAL_EXPERIMENT",
        "recovery_manifest": str(manifest_path),
        "recovery_manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint_sha256": _required_manifest_text(manifest, "source_checkpoint_sha256"),
        "source_metrics_sha256": _required_manifest_text(manifest, "source_metrics_sha256"),
        "active_prefix_preserved": True,
        "source_update": source_update,
        "end_update": loaded.state.update_idx,
        "source_global_step": source_global_step,
        "end_global_step": loaded.state.global_step,
        "source_elapsed_training_seconds": source_elapsed_float,
        "end_elapsed_training_seconds": loaded.state.elapsed_training_seconds,
        "source_logging_step": source_logging_step,
        "end_logging_step": loaded.state.logging_step,
        "new_training_record_count": len(new_training_records),
        "duplicate_logging_steps": False,
        "duplicate_logical_update_records": False,
        "optimizer_device": optimizer_device_evidence,
        "collector_league_trainer_checkpoint_validation": "PASS",
        "rng_restore": "PASS",
        "sample_legality_violations": metrics["mask/sample_legality_violations"],
        "mask_oracle_disagreements": metrics["mask/oracle_disagreements"],
        "all_metrics_finite": True,
        "stable_wandb_run_id": loaded.state.wandb_run_id,
        "checkpoint": str(latest_checkpoint),
        "checkpoint_sha256": loaded.evidence.sha256,
        "training_manifest": training_manifest,
        "audited_at": datetime.now(UTC).isoformat(),
    }
    audit_path = checked_run / "recovery" / "smoke-audit.json"
    report_path = checked_run / "recovery" / "smoke-audit-report.md"
    _atomic_write(
        audit_path,
        (json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    _atomic_write(
        report_path,
        (
            "# Recovery resume smoke audit\n\n"
            "Status: `PASS`\n\n"
            "Classification: bounded recovery smoke test; not final experiment evidence.\n\n"
            f"The checkpoint-aligned prefix remained byte-identical through byte {byte_end}. "
            f"Updates {source_update + 1} through {expected_end_update} resumed with contiguous "
            "logging, restored trainer/RNG state, correctly placed optimizer tensors, finite "
            "metrics, zero illegal sampled actions, and zero mask-oracle disagreements.\n"
        ).encode(),
    )
    return SmokeAuditResult(
        status="PASS",
        audit_path=audit_path,
        report_path=report_path,
        checkpoint_path=latest_checkpoint,
        end_update=loaded.state.update_idx,
        logging_step=loaded.state.logging_step,
    )
