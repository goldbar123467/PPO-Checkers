"""Low-overhead host, process, disk, and NVIDIA telemetry sampling."""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass

import psutil

NVIDIA_QUERY_FIELDS = (
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "fan.speed",
)


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    """One NVIDIA device sample with explicit unsupported values."""

    model: str
    utilization_percent: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    temperature_celsius: float | None
    power_draw_watts: float | None
    power_limit_watts: float | None
    core_clock_mhz: float | None
    memory_clock_mhz: float | None
    fan_speed_percent: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SystemTelemetry:
    """One non-blocking system and optional process telemetry sample."""

    sampled_monotonic_seconds: float
    cpu_total_percent: float
    cpu_per_core_percent: tuple[float, ...]
    ram_used_bytes: int
    ram_available_bytes: int
    process_pid: int | None
    process_cpu_percent: float | None
    process_rss_bytes: int | None
    process_read_bytes_per_second: float | None
    process_write_bytes_per_second: float | None
    gpu: GpuTelemetry | None

    def scalar_metrics(self) -> dict[str, float]:
        """Return finite scalar fields suitable for JSONL and W&B."""

        metrics = {
            "system/cpu_percent": self.cpu_total_percent,
            "system/ram_used_bytes": float(self.ram_used_bytes),
            "system/ram_available_bytes": float(self.ram_available_bytes),
        }
        optional = {
            "system/process_cpu_percent": self.process_cpu_percent,
            "system/process_rss_bytes": (
                None if self.process_rss_bytes is None else float(self.process_rss_bytes)
            ),
            "system/process_disk_read_bytes_per_second": self.process_read_bytes_per_second,
            "system/process_disk_write_bytes_per_second": self.process_write_bytes_per_second,
        }
        if self.gpu is not None:
            optional.update(
                {
                    "system/gpu_utilization_percent": self.gpu.utilization_percent,
                    "system/gpu_memory_used_mib": self.gpu.memory_used_mib,
                    "system/gpu_memory_total_mib": self.gpu.memory_total_mib,
                    "system/gpu_temperature_celsius": self.gpu.temperature_celsius,
                    "system/gpu_power_draw_watts": self.gpu.power_draw_watts,
                    "system/gpu_power_limit_watts": self.gpu.power_limit_watts,
                    "system/gpu_core_clock_mhz": self.gpu.core_clock_mhz,
                    "system/gpu_memory_clock_mhz": self.gpu.memory_clock_mhz,
                    "system/gpu_fan_speed_percent": self.gpu.fan_speed_percent,
                }
            )
        metrics.update(
            {
                name: float(value)
                for name, value in optional.items()
                if value is not None and math.isfinite(float(value))
            }
        )
        return metrics


def _optional_float(value: str) -> float | None:
    checked = value.strip()
    if not checked or checked.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        result = float(checked)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_nvidia_smi_csv(output: str) -> GpuTelemetry:
    """Parse the exact no-units single-device NVIDIA query used by this project."""

    if not isinstance(output, str):
        raise TypeError("NVIDIA output must be text")
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("NVIDIA query must return exactly one device row")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != len(NVIDIA_QUERY_FIELDS):
        raise ValueError("NVIDIA query returned an unexpected field count")
    model = values[0]
    if not model:
        raise ValueError("NVIDIA device model is empty")
    numeric = [_optional_float(value) for value in values[1:]]
    return GpuTelemetry(
        model=model,
        utilization_percent=numeric[0],
        memory_used_mib=numeric[1],
        memory_total_mib=numeric[2],
        temperature_celsius=numeric[3],
        power_draw_watts=numeric[4],
        power_limit_watts=numeric[5],
        core_clock_mhz=numeric[6],
        memory_clock_mhz=numeric[7],
        fan_speed_percent=numeric[8],
    )


def query_nvidia_smi() -> GpuTelemetry | None:
    """Query one NVIDIA device, returning explicit error telemetry when unavailable."""

    command = (
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return parse_nvidia_smi_csv(result.stdout)
    except FileNotFoundError:
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as error:
        return GpuTelemetry(
            model="UNKNOWN",
            utilization_percent=None,
            memory_used_mib=None,
            memory_total_mib=None,
            temperature_celsius=None,
            power_draw_watts=None,
            power_limit_watts=None,
            core_clock_mhz=None,
            memory_clock_mhz=None,
            fan_speed_percent=None,
            error=type(error).__name__,
        )


class SystemTelemetrySampler:
    """Stateful non-blocking sampler used by both training and read-only monitoring."""

    def __init__(self, *, process_pid: int | None = None) -> None:
        if process_pid is not None and (
            isinstance(process_pid, bool) or not isinstance(process_pid, int) or process_pid < 1
        ):
            raise ValueError("process_pid must be a positive integer or None")
        self._process_pid = process_pid
        self._process: psutil.Process | None = None
        self._previous_monotonic: float | None = None
        self._previous_read_bytes: int | None = None
        self._previous_write_bytes: int | None = None
        psutil.cpu_percent(interval=None, percpu=False)
        psutil.cpu_percent(interval=None, percpu=True)
        if process_pid is not None:
            try:
                self._process = psutil.Process(process_pid)
                self._process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None

    def sample(self) -> SystemTelemetry:
        """Collect one sample without sleeping or mutating process/system settings."""

        sampled = time.monotonic()
        total_cpu = float(psutil.cpu_percent(interval=None, percpu=False))
        per_core = tuple(float(value) for value in psutil.cpu_percent(interval=None, percpu=True))
        memory = psutil.virtual_memory()
        process_cpu: float | None = None
        process_rss: int | None = None
        read_rate: float | None = None
        write_rate: float | None = None
        if self._process is not None:
            try:
                process_cpu = float(self._process.cpu_percent(interval=None))
                process_rss = self._process.memory_info().rss
                counters = self._process.io_counters()
                if self._previous_monotonic is not None:
                    elapsed = sampled - self._previous_monotonic
                    if elapsed > 0.0:
                        if self._previous_read_bytes is not None:
                            read_rate = max(
                                0.0, (counters.read_bytes - self._previous_read_bytes) / elapsed
                            )
                        if self._previous_write_bytes is not None:
                            write_rate = max(
                                0.0,
                                (counters.write_bytes - self._previous_write_bytes) / elapsed,
                            )
                self._previous_read_bytes = counters.read_bytes
                self._previous_write_bytes = counters.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None
        self._previous_monotonic = sampled
        return SystemTelemetry(
            sampled_monotonic_seconds=sampled,
            cpu_total_percent=total_cpu,
            cpu_per_core_percent=per_core,
            ram_used_bytes=memory.used,
            ram_available_bytes=memory.available,
            process_pid=self._process_pid,
            process_cpu_percent=process_cpu,
            process_rss_bytes=process_rss,
            process_read_bytes_per_second=read_rate,
            process_write_bytes_per_second=write_rate,
            gpu=query_nvidia_smi(),
        )
