"""Read-only run monitor and system telemetry tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import yaml

import checkers.monitor as monitor_module
from checkers.config import RunConfig
from checkers.monitor import (
    MonitorSnapshot,
    collect_monitor_snapshot,
    live_runtime_process_id,
    read_metric_history_tolerant,
    render_monitor_snapshot,
)
from checkers.run_runtime import finish_runtime_state, new_runtime_state, write_runtime_state
from checkers.system_metrics import (
    GpuTelemetry,
    SystemTelemetry,
    SystemTelemetrySampler,
    parse_nvidia_smi_csv,
)

GPU_UTILIZATION_PERCENT = 88.0
GPU_MEMORY_TOTAL_MIB = 12227.0
GPU_POWER_DRAW_WATTS = 130.5
CURRENT_UPDATE = 10
ELAPSED_SECONDS = 40.0
REMAINING_SECONDS = 60.0
PERCENT_COMPLETE = 40.0


class FakeSampler:
    def __init__(self, sample: SystemTelemetry) -> None:
        self._sample = sample

    def sample(self) -> SystemTelemetry:
        return self._sample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _telemetry() -> SystemTelemetry:
    return SystemTelemetry(
        sampled_monotonic_seconds=10.0,
        cpu_total_percent=25.0,
        cpu_per_core_percent=(20.0, 30.0),
        ram_used_bytes=4 * 1024**3,
        ram_available_bytes=12 * 1024**3,
        process_pid=None,
        process_cpu_percent=50.0,
        process_rss_bytes=256 * 1024**2,
        process_read_bytes_per_second=1024.0,
        process_write_bytes_per_second=2048.0,
        gpu=GpuTelemetry(
            model="NVIDIA Test GPU",
            utilization_percent=80.0,
            memory_used_mib=2048.0,
            memory_total_mib=12227.0,
            temperature_celsius=55.0,
            power_draw_watts=120.0,
            power_limit_watts=250.0,
            core_clock_mhz=2500.0,
            memory_clock_mhz=14001.0,
            fan_speed_percent=None,
        ),
    )


def _run_directory(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "evaluations").mkdir()
    (run / "recovery").mkdir()
    config = RunConfig(
        experiment_id="monitor-unit",
        seed=17,
        device="cpu",
        total_timesteps=100,
        duration_seconds=100.0,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        eval_games=2,
    )
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8"
    )
    record = {
        "schema": "CHECKERS_METRIC_HISTORY_1",
        "logging_step": 0,
        "kind": "training",
        "update_idx": CURRENT_UPDATE,
        "global_step": CURRENT_UPDATE,
        "elapsed_training_seconds": ELAPSED_SECONDS,
        "metrics": {
            "train/policy_loss": -0.1,
            "train/value_loss": 0.2,
            "train/entropy": 0.3,
            "train/approx_kl": 0.01,
            "train/clipfrac": 0.05,
            "train/explained_variance": 0.7,
            "train/lr": 0.0003,
            "train/grad_norm": 0.4,
            "env/mean_game_len_moves": 60.0,
            "charts/SPS": 100.0,
            "mask/sample_legality_violations": 0.0,
            "mask/oracle_disagreements": 0.0,
        },
    }
    (run / "metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (run / "checkpoints" / "update-000010.pt").write_bytes(b"checkpoint")
    (run / "checkpoints" / "update-000010.pt.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    (run / "evaluations" / "update-000010-periodic.json").write_text(
        json.dumps({"update_idx": 10, "kind": "periodic"}) + "\n", encoding="utf-8"
    )
    (run / "recovery" / "recovery-manifest.json").write_text(
        json.dumps(
            {
                "final_recovery_status": "PREPARED",
                "current_commit": "abcdef0123456789",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = new_runtime_state(
        start_update=0,
        experiment_id=config.experiment_id,
        seed=config.seed,
        git_sha="abcdef0123456789",
        run_id="monitor-wandb-id",
        resume_from=None,
    )
    write_runtime_state(
        run / "runtime.json",
        finish_runtime_state(runtime, status="COMPLETED"),
    )
    return run


def test_nvidia_csv_parser_preserves_supported_and_unsupported_sensors() -> None:
    telemetry = parse_nvidia_smi_csv(
        "NVIDIA GeForce RTX 5070, 88, 4096, 12227, 61, 130.5, 250.0, 2800, 14001, [N/A]\n"
    )

    assert telemetry.model == "NVIDIA GeForce RTX 5070"
    assert telemetry.utilization_percent == GPU_UTILIZATION_PERCENT
    assert telemetry.memory_total_mib == GPU_MEMORY_TOTAL_MIB
    assert telemetry.power_draw_watts == GPU_POWER_DRAW_WATTS
    assert telemetry.fan_speed_percent is None


def test_metric_reader_tolerates_only_the_partial_final_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"kind":"training","update_idx":1}\n{"kind":"train')

    view = read_metric_history_tolerant(path)

    assert len(view.records) == 1
    assert view.records[0]["update_idx"] == 1
    assert view.warning is not None
    assert "partial final" in view.warning


def test_monitor_snapshot_is_read_only_and_labels_diagnostics(tmp_path: Path) -> None:
    run = _run_directory(tmp_path)
    before = {path: _sha256(path) for path in run.rglob("*") if path.is_file()}
    sampler = cast(SystemTelemetrySampler, FakeSampler(_telemetry()))

    snapshot = collect_monitor_snapshot(run_directory=run, sampler=sampler)
    rendered = render_monitor_snapshot(snapshot)

    after = {path: _sha256(path) for path in run.rglob("*") if path.is_file()}
    assert after == before
    assert snapshot.run_status == "FINISHED"
    assert snapshot.recovery_status == "PREPARED"
    assert snapshot.current_update == CURRENT_UPDATE
    assert snapshot.transitions_completed == CURRENT_UPDATE
    assert snapshot.elapsed_training_seconds == ELAPSED_SECONDS
    assert snapshot.remaining_training_seconds == REMAINING_SECONDS
    assert snapshot.percentage_complete == PERCENT_COMPLETE
    assert snapshot.latest_checkpoint is not None
    assert snapshot.latest_checkpoint.update_idx == CURRENT_UPDATE
    assert snapshot.latest_evaluation is not None
    assert snapshot.latest_evaluation.evidence_class == "DIAGNOSTIC_ONLY"
    assert snapshot.ppo_metrics["reward_mean"] is None
    assert snapshot.ppo_metrics["nan_or_infinity_count"] == 0.0
    assert "DIAGNOSTIC_ONLY" in rendered
    assert "reward mean / episode length mean: N/A" in rendered
    assert "NVIDIA Test GPU" in rendered


def test_monitor_status_distinguishes_running_idle_crashed_finished_and_stopped() -> None:
    runtime = new_runtime_state(
        start_update=0,
        experiment_id="status-unit",
        seed=1,
        git_sha="abc",
        run_id="id",
        resume_from=None,
    )
    telemetry = _telemetry()
    idle_gpu = replace(cast(GpuTelemetry, telemetry.gpu), utilization_percent=0.0)
    idle_telemetry = replace(telemetry, process_cpu_percent=0.0, gpu=idle_gpu)

    assert monitor_module._process_alive(runtime)
    assert not monitor_module._process_alive(
        replace(runtime, process_start_ticks=cast(int, runtime.process_start_ticks) + 1)
    )

    assert (
        monitor_module._run_status(
            runtime=runtime,
            process_alive=True,
            telemetry=telemetry,
            metric_age=1.0,
            completed_manifest=None,
        )
        == "RUNNING"
    )
    assert (
        monitor_module._run_status(
            runtime=runtime,
            process_alive=True,
            telemetry=idle_telemetry,
            metric_age=120.0,
            completed_manifest=None,
        )
        == "IDLE_OR_WAITING"
    )
    assert (
        monitor_module._run_status(
            runtime=runtime,
            process_alive=False,
            telemetry=telemetry,
            metric_age=1.0,
            completed_manifest=None,
        )
        == "CRASHED"
    )
    assert (
        monitor_module._run_status(
            runtime=finish_runtime_state(runtime, status="FAILED"),
            process_alive=False,
            telemetry=telemetry,
            metric_age=None,
            completed_manifest=None,
        )
        == "CRASHED"
    )
    assert (
        monitor_module._run_status(
            runtime=finish_runtime_state(runtime, status="COMPLETED"),
            process_alive=False,
            telemetry=telemetry,
            metric_age=None,
            completed_manifest=None,
        )
        == "FINISHED"
    )
    assert (
        monitor_module._run_status(
            runtime=None,
            process_alive=False,
            telemetry=telemetry,
            metric_age=None,
            completed_manifest={"status": "completed"},
        )
        == "FINISHED"
    )
    assert (
        monitor_module._run_status(
            runtime=None,
            process_alive=False,
            telemetry=telemetry,
            metric_age=None,
            completed_manifest=None,
        )
        == "STOPPED"
    )


def test_monitor_handles_empty_run_corrupt_runtime_and_invalid_evaluations(tmp_path: Path) -> None:
    run = tmp_path / "empty-run"
    run.mkdir()
    config = RunConfig(
        experiment_id="empty-monitor",
        seed=2,
        device="cpu",
        total_timesteps=2,
        duration_seconds=None,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        eval_games=2,
    )
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8"
    )
    (run / "runtime.json").write_text("{", encoding="utf-8")
    evaluations = run / "evaluations"
    evaluations.mkdir()
    (evaluations / "update-000001-periodic.json").write_text("{", encoding="utf-8")
    (evaluations / "update-000002-final.json").write_text(
        json.dumps({"update_idx": 2, "kind": "final"}), encoding="utf-8"
    )
    sampler = cast(SystemTelemetrySampler, FakeSampler(_telemetry()))

    snapshot = collect_monitor_snapshot(run_directory=run, sampler=sampler)

    assert snapshot.run_status == "STOPPED"
    assert snapshot.current_update is None
    assert snapshot.remaining_training_seconds is None
    assert snapshot.latest_checkpoint is None
    assert snapshot.latest_evaluation is not None
    assert snapshot.latest_evaluation.evidence_class == "FINAL_POWERED"
    assert snapshot.latest_warning_or_error == "runtime state is unreadable"
    assert snapshot.ppo_metrics["policy_loss"] is None
    assert live_runtime_process_id(run) is None
    with pytest.raises(TypeError, match="run_directory"):
        collect_monitor_snapshot(run_directory=cast(Path, "bad"), sampler=sampler)
    with pytest.raises(TypeError, match="run_directory"):
        live_runtime_process_id(cast(Path, "bad"))
    with pytest.raises(TypeError, match="snapshot"):
        render_monitor_snapshot(cast(MonitorSnapshot, "bad"))


def test_metric_reader_reports_nonfinal_corruption_and_invalid_mapping(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(b"{\n{}\n")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")

    corrupt_view = read_metric_history_tolerant(corrupt)
    invalid_view = read_metric_history_tolerant(invalid)

    assert corrupt_view.records == ()
    assert corrupt_view.warning is not None
    assert "corrupt" in corrupt_view.warning
    assert invalid_view.records == ()
    assert invalid_view.warning == "invalid metrics mapping at line 1"
