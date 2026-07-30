"""Process and NVIDIA device monitoring with graceful capability fallback."""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from time import monotonic
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One process/system/GPU monitoring sample."""

    process_rss_mb: float | None
    process_cpu_percent: float | None
    system_memory_percent: float | None
    gpu_memory_used_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Latest values and peaks collected during a replay run."""

    samples: int
    latest_process_cpu_percent: float | None
    latest_system_memory_percent: float | None
    peak_process_rss_mb: float | None
    peak_gpu_memory_used_mb: float | None
    peak_gpu_utilization_percent: float | None
    peak_gpu_temperature_c: float | None
    peak_gpu_power_w: float | None
    sampling_failures: int
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MetricsProvider(Protocol):
    """Provider contract used by the periodic monitor."""

    def sample(self) -> ResourceSample: ...


class SystemMetricsProvider:
    """Collect process metrics through psutil and GPU metrics through nvidia-smi."""

    def __init__(self) -> None:
        self._process: Any | None = None
        self._psutil: Any | None = None
        self._psutil_checked = False
        self._gpu_available: bool | None = None

    def sample(self) -> ResourceSample:
        self._ensure_psutil()
        process_rss_mb: float | None = None
        process_cpu_percent: float | None = None
        system_memory_percent: float | None = None
        if self._process is not None and self._psutil is not None:
            process_rss_mb = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
            process_cpu_percent = float(self._process.cpu_percent(interval=None))
            system_memory_percent = float(self._psutil.virtual_memory().percent)

        gpu = self._sample_nvidia()
        return ResourceSample(
            process_rss_mb=process_rss_mb,
            process_cpu_percent=process_cpu_percent,
            system_memory_percent=system_memory_percent,
            gpu_memory_used_mb=gpu[0],
            gpu_utilization_percent=gpu[1],
            gpu_temperature_c=gpu[2],
            gpu_power_w=gpu[3],
        )

    def _ensure_psutil(self) -> None:
        if self._psutil_checked:
            return
        self._psutil_checked = True
        try:
            self._psutil = importlib.import_module("psutil")
            self._process = self._psutil.Process()
        except ModuleNotFoundError:
            self._psutil = None
            self._process = None

    def _sample_nvidia(self) -> tuple[float | None, float | None, float | None, float | None]:
        if self._gpu_available is False:
            return None, None, None, None
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            self._gpu_available = False
            return None, None, None, None
        if completed.returncode != 0 or not completed.stdout.strip():
            self._gpu_available = False
            return None, None, None, None

        self._gpu_available = True
        first_gpu = completed.stdout.splitlines()[0]
        values = [_parse_optional_float(value) for value in first_gpu.split(",")]
        values.extend([None] * (4 - len(values)))
        return values[0], values[1], values[2], values[3]


class RuntimeResourceMonitor:
    """Periodically sample resources and retain only aggregate state."""

    def __init__(
        self,
        *,
        provider: MetricsProvider | None = None,
        interval_seconds: float = 1.0,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self._provider = provider or SystemMetricsProvider()
        self._interval_seconds = interval_seconds
        self._monotonic_clock = monotonic_clock
        self._lock = Lock()
        self._last_sampled_at: float | None = None
        self._samples = 0
        self._latest: ResourceSample | None = None
        self._peak_process_rss_mb: float | None = None
        self._peak_gpu_memory_used_mb: float | None = None
        self._peak_gpu_utilization_percent: float | None = None
        self._peak_gpu_temperature_c: float | None = None
        self._peak_gpu_power_w: float | None = None
        self._sampling_failures = 0
        self._last_error: str | None = None

    def observe(self, *, force: bool = False) -> ResourceSample | None:
        """Sample when the interval has elapsed and update peak counters."""

        now = float(self._monotonic_clock())
        with self._lock:
            if not force and self._last_sampled_at is not None and now - self._last_sampled_at < self._interval_seconds:
                return None
            self._last_sampled_at = now
        try:
            sample = self._provider.sample()
        except Exception as exc:
            with self._lock:
                self._sampling_failures += 1
                self._last_error = repr(exc)
            return None

        with self._lock:
            self._samples += 1
            self._latest = sample
            self._last_error = None
            self._peak_process_rss_mb = _maximum(self._peak_process_rss_mb, sample.process_rss_mb)
            self._peak_gpu_memory_used_mb = _maximum(self._peak_gpu_memory_used_mb, sample.gpu_memory_used_mb)
            self._peak_gpu_utilization_percent = _maximum(
                self._peak_gpu_utilization_percent,
                sample.gpu_utilization_percent,
            )
            self._peak_gpu_temperature_c = _maximum(self._peak_gpu_temperature_c, sample.gpu_temperature_c)
            self._peak_gpu_power_w = _maximum(self._peak_gpu_power_w, sample.gpu_power_w)
        return sample

    def summary(self) -> ResourceSummary:
        with self._lock:
            latest = self._latest
            return ResourceSummary(
                samples=self._samples,
                latest_process_cpu_percent=None if latest is None else latest.process_cpu_percent,
                latest_system_memory_percent=None if latest is None else latest.system_memory_percent,
                peak_process_rss_mb=self._peak_process_rss_mb,
                peak_gpu_memory_used_mb=self._peak_gpu_memory_used_mb,
                peak_gpu_utilization_percent=self._peak_gpu_utilization_percent,
                peak_gpu_temperature_c=self._peak_gpu_temperature_c,
                peak_gpu_power_w=self._peak_gpu_power_w,
                sampling_failures=self._sampling_failures,
                last_error=self._last_error,
            )


def _maximum(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def _parse_optional_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None
