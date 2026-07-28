"""Host and NVIDIA telemetry failure-mode tests."""

from __future__ import annotations

import os
import subprocess
from typing import Never, cast

import pytest

from checkers import system_metrics
from checkers.system_metrics import (
    GpuTelemetry,
    SystemTelemetry,
    SystemTelemetrySampler,
    parse_nvidia_smi_csv,
    query_nvidia_smi,
)

PARSED_MEMORY_TOTAL_MIB = 3.0
GPU_UTILIZATION_PERCENT = 50.0


def test_system_scalar_metrics_omit_unsupported_and_nonfinite_values() -> None:
    sample = SystemTelemetry(
        sampled_monotonic_seconds=1.0,
        cpu_total_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_used_bytes=100,
        ram_available_bytes=200,
        process_pid=None,
        process_cpu_percent=float("nan"),
        process_rss_bytes=None,
        process_read_bytes_per_second=None,
        process_write_bytes_per_second=None,
        gpu=None,
    )

    metrics = sample.scalar_metrics()

    assert metrics == {
        "system/cpu_percent": 10.0,
        "system/ram_used_bytes": 100.0,
        "system/ram_available_bytes": 200.0,
    }


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("", "exactly one"),
        ("a\nb\n", "exactly one"),
        ("GPU,1\n", "field count"),
        (",1,2,3,4,5,6,7,8,9\n", "model is empty"),
    ],
)
def test_nvidia_parser_rejects_malformed_rows(output: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_nvidia_smi_csv(output)
    with pytest.raises(TypeError, match="must be text"):
        parse_nvidia_smi_csv(cast(str, 1))


def test_nvidia_parser_maps_bad_and_nonfinite_sensors_to_unavailable() -> None:
    parsed = parse_nvidia_smi_csv("GPU,bad,nan,3,4,5,6,7,8,not supported\n")

    assert parsed.utilization_percent is None
    assert parsed.memory_used_mib is None
    assert parsed.memory_total_mib == PARSED_MEMORY_TOTAL_MIB
    assert parsed.fan_speed_percent is None


def test_nvidia_query_handles_missing_binary_and_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args: object, **_kwargs: object) -> Never:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)
    assert query_nvidia_smi() is None

    def failed(*_args: object, **_kwargs: object) -> Never:
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(subprocess, "run", failed)
    telemetry = query_nvidia_smi()
    assert telemetry is not None
    assert telemetry.model == "UNKNOWN"
    assert telemetry.error == "CalledProcessError"


def test_stateful_sampler_reports_process_io_rates_on_second_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_metrics, "query_nvidia_smi", lambda: None)
    sampler = SystemTelemetrySampler(process_pid=os.getpid())

    first = sampler.sample()
    second = sampler.sample()

    assert first.process_pid == os.getpid()
    assert first.process_rss_bytes is not None
    assert second.process_read_bytes_per_second is not None
    assert second.process_write_bytes_per_second is not None
    assert second.gpu is None
    assert "system/process_rss_bytes" in second.scalar_metrics()
    with pytest.raises(ValueError, match="process_pid"):
        SystemTelemetrySampler(process_pid=0)


def test_gpu_scalar_metrics_include_only_finite_supported_sensors() -> None:
    gpu = GpuTelemetry(
        model="GPU",
        utilization_percent=50.0,
        memory_used_mib=100.0,
        memory_total_mib=200.0,
        temperature_celsius=60.0,
        power_draw_watts=70.0,
        power_limit_watts=80.0,
        core_clock_mhz=90.0,
        memory_clock_mhz=100.0,
        fan_speed_percent=None,
    )
    sample = SystemTelemetry(
        sampled_monotonic_seconds=1.0,
        cpu_total_percent=1.0,
        cpu_per_core_percent=(),
        ram_used_bytes=1,
        ram_available_bytes=2,
        process_pid=None,
        process_cpu_percent=None,
        process_rss_bytes=None,
        process_read_bytes_per_second=None,
        process_write_bytes_per_second=None,
        gpu=gpu,
    )

    metrics = sample.scalar_metrics()

    assert metrics["system/gpu_utilization_percent"] == GPU_UTILIZATION_PERCENT
    assert "system/gpu_fan_speed_percent" not in metrics
