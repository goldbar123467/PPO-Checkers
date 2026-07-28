"""Read-only terminal monitoring for local Checkers PPO runs."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import psutil

from checkers.config import RunConfig, load_run_config
from checkers.run_runtime import RuntimeState, read_process_start_ticks, read_runtime_state
from checkers.system_metrics import SystemTelemetry, SystemTelemetrySampler

CHECKPOINT_PATTERN = re.compile(r"^update-(\d{6})\.pt$")
STALE_SECONDS = 30.0
IDLE_SECONDS = 60.0
LEGACY_PROCESS_START_TOLERANCE_SECONDS = 30.0
IDLE_GPU_UTILIZATION_PERCENT = 5.0


@dataclass(frozen=True, slots=True)
class MetricHistoryView:
    """Valid JSONL prefix plus any explicitly tolerated tail problem."""

    records: tuple[dict[str, object], ...]
    warning: str | None


@dataclass(frozen=True, slots=True)
class EvaluationView:
    """Latest persisted evaluation classification."""

    update_idx: int
    kind: str
    evidence_class: str


@dataclass(frozen=True, slots=True)
class CheckpointView:
    """Latest checkpoint with both data and digest files visible."""

    update_idx: int
    path: Path
    age_seconds: float


@dataclass(frozen=True, slots=True)
class MonitorSnapshot:
    """One complete read-only monitor snapshot."""

    run_id: str | None
    experiment_id: str
    seed: int
    commit: str | None
    run_status: str
    recovery_status: str | None
    process_id: int | None
    process_uptime_seconds: float | None
    current_update: int | None
    latest_checkpoint: CheckpointView | None
    transitions_completed: int | None
    elapsed_training_seconds: float | None
    remaining_training_seconds: float | None
    percentage_complete: float | None
    time_since_last_metric_seconds: float | None
    latest_evaluation: EvaluationView | None
    latest_warning_or_error: str | None
    ppo_metrics: dict[str, float | None]
    telemetry: SystemTelemetry
    disk_free_bytes: int
    metrics_stale: bool


def read_metric_history_tolerant(path: Path) -> MetricHistoryView:
    """Read a valid JSONL prefix while tolerating a partially written final line."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.exists():
        return MetricHistoryView(records=(), warning=None)
    try:
        payload = path.read_bytes()
    except OSError as error:
        return MetricHistoryView(records=(), warning=f"metrics unreadable: {type(error).__name__}")
    lines = payload.splitlines(keepends=True)
    records: list[dict[str, object]] = []
    warning: str | None = None
    for index, line in enumerate(lines):
        encoded = line[:-1] if line.endswith(b"\n") else line
        if encoded.endswith(b"\r"):
            encoded = encoded[:-1]
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            position = "partial final" if index == len(lines) - 1 else "corrupt"
            warning = f"{position} metrics line {index + 1}: {type(error).__name__}"
            break
        if not isinstance(value, dict):
            warning = f"invalid metrics mapping at line {index + 1}"
            break
        records.append(cast(dict[str, object], value))
    return MetricHistoryView(records=tuple(records), warning=warning)


def _record_int(record: dict[str, object], name: str) -> int | None:
    value = record.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _record_float(record: dict[str, object], name: str) -> float | None:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    checked = float(value)
    return checked if math.isfinite(checked) and checked >= 0.0 else None


def _latest_training_record(records: tuple[dict[str, object], ...]) -> dict[str, object] | None:
    return next(
        (record for record in reversed(records) if record.get("kind") == "training"),
        None,
    )


def _latest_checkpoint(run_directory: Path, now: float) -> CheckpointView | None:
    checkpoints: list[CheckpointView] = []
    for path in (run_directory / "checkpoints").glob("update-*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        sidecar = path.with_suffix(f"{path.suffix}.sha256")
        try:
            if match is None or not path.is_file() or not sidecar.is_file():
                continue
            age = max(0.0, now - max(path.stat().st_mtime, sidecar.stat().st_mtime))
        except OSError:
            continue
        checkpoints.append(
            CheckpointView(update_idx=int(match.group(1)), path=path, age_seconds=age)
        )
    return max(checkpoints, key=lambda item: item.update_idx, default=None)


def _latest_evaluation(run_directory: Path) -> EvaluationView | None:
    evaluations: list[EvaluationView] = []
    for path in (run_directory / "evaluations").glob("update-*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        update_idx = value.get("update_idx")
        kind = value.get("kind")
        if (
            isinstance(update_idx, bool)
            or not isinstance(update_idx, int)
            or update_idx < 0
            or not isinstance(kind, str)
        ):
            continue
        evaluations.append(
            EvaluationView(
                update_idx=update_idx,
                kind=kind,
                evidence_class="FINAL_POWERED" if kind == "final" else "DIAGNOSTIC_ONLY",
            )
        )
    return max(
        evaluations,
        key=lambda item: (item.update_idx, item.kind == "final"),
        default=None,
    )


def _read_recovery(run_directory: Path) -> tuple[str | None, str | None]:
    path = run_directory / "recovery" / "recovery-manifest.json"
    if not path.exists():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID", None
    if not isinstance(value, dict):
        return "INVALID", None
    status = value.get("final_recovery_status")
    commit = value.get("current_commit")
    return (
        status if isinstance(status, str) else "INVALID",
        commit if isinstance(commit, str) else None,
    )


def _latest_completed_manifest(run_directory: Path) -> dict[str, object] | None:
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
    return max(candidates, key=lambda item: item[0], default=(0, None))[1]


def _process_alive(runtime: RuntimeState | None) -> bool:
    if runtime is None:
        return False
    try:
        process = psutil.Process(runtime.pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if runtime.process_start_ticks is not None:
            return read_process_start_ticks(runtime.pid) == runtime.process_start_ticks
        started = datetime.fromisoformat(runtime.started_at).timestamp()
        return abs(process.create_time() - started) < LEGACY_PROCESS_START_TOLERANCE_SECONDS
    except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def live_runtime_process_id(run_directory: Path) -> int | None:
    """Return the live, creation-time-matched trainer PID, if one exists."""

    if not isinstance(run_directory, Path):
        raise TypeError("run_directory must be a Path")
    try:
        runtime = read_runtime_state(run_directory / "runtime.json")
    except ValueError:
        return None
    return runtime.pid if runtime is not None and _process_alive(runtime) else None


def _run_status(  # noqa: PLR0911
    *,
    runtime: RuntimeState | None,
    process_alive: bool,
    telemetry: SystemTelemetry,
    metric_age: float | None,
    completed_manifest: dict[str, object] | None,
) -> str:
    if runtime is not None and runtime.status == "COMPLETED":
        return "FINISHED"
    if runtime is not None and runtime.status == "FAILED":
        return "CRASHED"
    if runtime is not None and runtime.status == "RUNNING" and not process_alive:
        return "CRASHED"
    if process_alive:
        gpu_utilization = None if telemetry.gpu is None else telemetry.gpu.utilization_percent
        low_process_cpu = (telemetry.process_cpu_percent or 0.0) < 1.0
        low_gpu = (gpu_utilization or 0.0) < IDLE_GPU_UTILIZATION_PERCENT
        if metric_age is not None and metric_age > IDLE_SECONDS and low_process_cpu and low_gpu:
            return "IDLE_OR_WAITING"
        return "RUNNING"
    if completed_manifest is not None and completed_manifest.get("status") == "completed":
        return "FINISHED"
    return "STOPPED"


def _ppo_metrics(
    training_record: dict[str, object] | None,
    *,
    batch_size: int,
) -> dict[str, float | None]:
    names = {
        "policy_loss": "train/policy_loss",
        "value_loss": "train/value_loss",
        "entropy": "train/entropy",
        "approximate_kl": "train/approx_kl",
        "clip_fraction": "train/clipfrac",
        "explained_variance": "train/explained_variance",
        "learning_rate": "train/lr",
        "gradient_norm": "train/grad_norm",
        "episode_length_mean": "env/mean_game_len_moves",
        "rollout_throughput": "charts/SPS",
        "transitions_per_second": "charts/SPS",
        "illegal_sampled_actions": "mask/sample_legality_violations",
        "mask_oracle_disagreements": "mask/oracle_disagreements",
    }
    output: dict[str, float | None] = {name: None for name in names}
    output["reward_mean"] = None
    output["updates_per_second"] = None
    output["nan_or_infinity_count"] = None
    if training_record is None:
        return output
    raw_metrics = training_record.get("metrics")
    if not isinstance(raw_metrics, dict):
        return output
    metrics = cast(dict[str, object], raw_metrics)
    for output_name, metric_name in names.items():
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            checked = float(value)
            output[output_name] = checked if math.isfinite(checked) else None
    throughput = output["transitions_per_second"]
    if throughput is not None:
        output["updates_per_second"] = throughput / batch_size
    output["nan_or_infinity_count"] = float(
        sum(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
            for value in metrics.values()
        )
    )
    return output


def collect_monitor_snapshot(  # noqa: PLR0915
    *,
    run_directory: Path,
    sampler: SystemTelemetrySampler | None = None,
) -> MonitorSnapshot:
    """Collect one read-only snapshot from local artifacts and live telemetry."""

    if not isinstance(run_directory, Path):
        raise TypeError("run_directory must be a Path")
    checked_run = run_directory.resolve(strict=True)
    if not checked_run.is_dir():
        raise ValueError("run_directory must be an existing directory")
    config: RunConfig = load_run_config(
        (checked_run / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    runtime_warning: str | None
    try:
        runtime = read_runtime_state(checked_run / "runtime.json")
    except ValueError as error:
        runtime = None
        runtime_warning = str(error)
    else:
        runtime_warning = None
    alive = _process_alive(runtime)
    checked_sampler = sampler
    if checked_sampler is None:
        checked_sampler = SystemTelemetrySampler(
            process_pid=runtime.pid if runtime is not None and alive else None
        )
    telemetry = checked_sampler.sample()
    history = read_metric_history_tolerant(checked_run / "metrics.jsonl")
    training_record = _latest_training_record(history.records)
    latest_record = history.records[-1] if history.records else None
    now = time.time()
    try:
        metric_age = max(0.0, now - (checked_run / "metrics.jsonl").stat().st_mtime)
    except OSError:
        metric_age = None
    checkpoint = _latest_checkpoint(checked_run, now)
    evaluation = _latest_evaluation(checked_run)
    recovery_status, recovery_commit = _read_recovery(checked_run)
    completed_manifest = _latest_completed_manifest(checked_run)
    status = _run_status(
        runtime=runtime,
        process_alive=alive,
        telemetry=telemetry,
        metric_age=metric_age,
        completed_manifest=completed_manifest,
    )
    current_update = None if latest_record is None else _record_int(latest_record, "update_idx")
    transitions = None if latest_record is None else _record_int(latest_record, "global_step")
    elapsed = (
        None if latest_record is None else _record_float(latest_record, "elapsed_training_seconds")
    )
    remaining = (
        None
        if elapsed is None or config.duration_seconds is None
        else max(0.0, config.duration_seconds - elapsed)
    )
    percentage = (
        None
        if elapsed is None or config.duration_seconds is None
        else min(100.0, 100.0 * elapsed / config.duration_seconds)
    )
    uptime: float | None = None
    if runtime is not None:
        try:
            uptime = max(
                0.0,
                datetime.now(UTC).timestamp()
                - datetime.fromisoformat(runtime.started_at).timestamp(),
            )
        except ValueError:
            runtime_warning = "runtime start timestamp is invalid"
    latest_problem = runtime_warning or history.warning
    if runtime is not None:
        latest_problem = runtime.latest_error or runtime.latest_warning or latest_problem
    manifest_commit: str | None = None
    if completed_manifest is not None:
        raw_manifest_commit = completed_manifest.get("git_sha")
        if isinstance(raw_manifest_commit, str):
            manifest_commit = raw_manifest_commit
    commit: str | None = (
        runtime.git_sha if runtime is not None else recovery_commit or manifest_commit
    )
    return MonitorSnapshot(
        run_id=None if runtime is None else runtime.run_id,
        experiment_id=config.experiment_id,
        seed=config.seed,
        commit=commit,
        run_status=status,
        recovery_status=recovery_status,
        process_id=None if runtime is None else runtime.pid,
        process_uptime_seconds=uptime,
        current_update=current_update,
        latest_checkpoint=checkpoint,
        transitions_completed=transitions,
        elapsed_training_seconds=elapsed,
        remaining_training_seconds=remaining,
        percentage_complete=percentage,
        time_since_last_metric_seconds=metric_age,
        latest_evaluation=evaluation,
        latest_warning_or_error=latest_problem,
        ppo_metrics=_ppo_metrics(training_record, batch_size=config.batch_size),
        telemetry=telemetry,
        disk_free_bytes=shutil.disk_usage(checked_run).free,
        metrics_stale=metric_age is not None and metric_age > STALE_SECONDS,
    )


def _format_number(value: float | int | None, *, precision: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{precision}f}"


def _format_duration(value: float | None) -> str:
    if value is None:
        return "N/A"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "N/A"
    gibibytes = value / (1024**3)
    return f"{gibibytes:.2f} GiB"


def _format_unit(value: float | int | None, unit: str, *, precision: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{_format_number(value, precision=precision)}{unit}"


def render_monitor_snapshot(snapshot: MonitorSnapshot) -> str:
    """Render one compact terminal snapshot without side effects."""

    if not isinstance(snapshot, MonitorSnapshot):
        raise TypeError("snapshot must be a MonitorSnapshot")
    checkpoint = snapshot.latest_checkpoint
    evaluation = snapshot.latest_evaluation
    gpu = snapshot.telemetry.gpu
    ppo = snapshot.ppo_metrics
    per_core = ", ".join(f"{value:.1f}%" for value in snapshot.telemetry.cpu_per_core_percent)
    lines = [
        "Checkers PPO run monitor (read only)",
        "",
        "Run",
        f"  experiment: {snapshot.experiment_id}",
        f"  run ID: {snapshot.run_id or 'N/A'}",
        f"  seed / commit: {snapshot.seed} / {snapshot.commit or 'N/A'}",
        f"  status / recovery: {snapshot.run_status} / {snapshot.recovery_status or 'N/A'}",
        f"  PID / uptime: {_format_number(snapshot.process_id)} / "
        f"{_format_duration(snapshot.process_uptime_seconds)}",
        f"  update / checkpoint: {_format_number(snapshot.current_update)} / "
        f"{('N/A' if checkpoint is None else checkpoint.path.name)}",
        f"  transitions: {_format_number(snapshot.transitions_completed)}",
        f"  timed training: {_format_duration(snapshot.elapsed_training_seconds)} elapsed, "
        f"{_format_duration(snapshot.remaining_training_seconds)} remaining, "
        f"{_format_number(snapshot.percentage_complete, precision=1)}%",
        f"  last metric / checkpoint age: "
        f"{_format_duration(snapshot.time_since_last_metric_seconds)} / "
        f"{_format_duration(None if checkpoint is None else checkpoint.age_seconds)}",
        f"  metrics stale: {snapshot.metrics_stale}",
        f"  evaluation: "
        f"{
            (
                'N/A'
                if evaluation is None
                else f'{evaluation.kind} '
                f'({evaluation.evidence_class}, update {evaluation.update_idx})'
            )
        }",
        f"  latest warning/error: {snapshot.latest_warning_or_error or 'N/A'}",
        "",
        "PPO (latest training record)",
        f"  policy/value loss: {_format_number(ppo['policy_loss'])} / "
        f"{_format_number(ppo['value_loss'])}",
        f"  entropy / approx KL / clip fraction: {_format_number(ppo['entropy'])} / "
        f"{_format_number(ppo['approximate_kl'])} / {_format_number(ppo['clip_fraction'])}",
        f"  explained variance / LR / grad norm: "
        f"{_format_number(ppo['explained_variance'])} / "
        f"{_format_number(ppo['learning_rate'], precision=6)} / "
        f"{_format_number(ppo['gradient_norm'])}",
        f"  reward mean / episode length mean: {_format_number(ppo['reward_mean'])} / "
        f"{_format_number(ppo['episode_length_mean'])}",
        f"  rollout / update / transition rate: {_format_number(ppo['rollout_throughput'])} / "
        f"{_format_number(ppo['updates_per_second'])} / "
        f"{_format_number(ppo['transitions_per_second'])}",
        f"  illegal actions / oracle disagreements / non-finite: "
        f"{_format_number(ppo['illegal_sampled_actions'])} / "
        f"{_format_number(ppo['mask_oracle_disagreements'])} / "
        f"{_format_number(ppo['nan_or_infinity_count'])}",
        "",
        "Hardware",
        f"  CPU total: {snapshot.telemetry.cpu_total_percent:.1f}%",
        f"  CPU per core: {per_core or 'N/A'}",
        f"  process CPU / RSS: "
        f"{_format_unit(snapshot.telemetry.process_cpu_percent, '%')} / "
        f"{_format_bytes(snapshot.telemetry.process_rss_bytes)}",
        f"  RAM used / available: {_format_bytes(snapshot.telemetry.ram_used_bytes)} / "
        f"{_format_bytes(snapshot.telemetry.ram_available_bytes)}",
        f"  GPU: {('N/A' if gpu is None else gpu.model)}",
        f"  GPU utilization / memory: "
        f"{_format_unit(None if gpu is None else gpu.utilization_percent, '%')} / "
        f"{_format_number(None if gpu is None else gpu.memory_used_mib, precision=0)} MiB of "
        f"{_format_number(None if gpu is None else gpu.memory_total_mib, precision=0)} MiB",
        f"  GPU temp / power: "
        f"{_format_number(None if gpu is None else gpu.temperature_celsius, precision=1)} C / "
        f"{_format_number(None if gpu is None else gpu.power_draw_watts, precision=1)} W of "
        f"{_format_number(None if gpu is None else gpu.power_limit_watts, precision=1)} W",
        f"  GPU core/memory clock / fan: "
        f"{_format_number(None if gpu is None else gpu.core_clock_mhz, precision=0)} / "
        f"{_format_number(None if gpu is None else gpu.memory_clock_mhz, precision=0)} MHz / "
        f"{_format_unit(None if gpu is None else gpu.fan_speed_percent, '%')}",
        f"  disk free: {_format_bytes(snapshot.disk_free_bytes)}",
        f"  process disk read/write: "
        f"{_format_number(snapshot.telemetry.process_read_bytes_per_second)} / "
        f"{_format_number(snapshot.telemetry.process_write_bytes_per_second)} bytes/s",
    ]
    return "\n".join(lines)
