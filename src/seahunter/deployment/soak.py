"""Streaming soak-test and fault-recovery acceptance accounting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


@dataclass(frozen=True, slots=True)
class SoakSample:
    elapsed_seconds: float
    process_rss_mb: float
    gpu_memory_mb: float | None
    p95_latency_ms: float
    frames_processed: int
    recoverable_errors: int = 0
    fatal_errors: int = 0

    def __post_init__(self) -> None:
        numeric = (self.elapsed_seconds, self.process_rss_mb, self.p95_latency_ms)
        if any(not isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("soak sample values must be finite and non-negative")
        if self.gpu_memory_mb is not None and (not isfinite(self.gpu_memory_mb) or self.gpu_memory_mb < 0.0):
            raise ValueError("soak GPU memory must be finite and non-negative")
        if self.frames_processed < 0 or self.recoverable_errors < 0 or self.fatal_errors < 0:
            raise ValueError("soak counters must be non-negative")


@dataclass(frozen=True, slots=True)
class SoakReport:
    samples: int
    duration_hours: float
    memory_slope_mb_per_hour: float
    peak_process_rss_mb: float
    peak_gpu_memory_mb: float | None
    peak_p95_latency_ms: float
    frames_processed: int
    recoverable_errors: int
    fatal_errors: int

    def passes(self, *, required_hours: float, maximum_memory_slope_mb_per_hour: float) -> bool:
        return (
            self.duration_hours >= required_hours
            and self.memory_slope_mb_per_hour <= maximum_memory_slope_mb_per_hour
            and self.fatal_errors == 0
        )


class SoakAccumulator:
    """Retain O(1) aggregate state regardless of soak duration."""

    def __init__(self) -> None:
        self._samples = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xx = 0.0
        self._sum_xy = 0.0
        self._duration_seconds = 0.0
        self._last_elapsed = -1.0
        self._peak_rss = 0.0
        self._peak_gpu: float | None = None
        self._peak_latency = 0.0
        self._frames = 0
        self._recoverable = 0
        self._fatal = 0

    def add(self, sample: SoakSample) -> None:
        if sample.elapsed_seconds <= self._last_elapsed:
            raise ValueError("soak elapsed_seconds must increase strictly")
        self._last_elapsed = sample.elapsed_seconds
        hours = sample.elapsed_seconds / 3600.0
        self._samples += 1
        self._sum_x += hours
        self._sum_y += sample.process_rss_mb
        self._sum_xx += hours * hours
        self._sum_xy += hours * sample.process_rss_mb
        self._duration_seconds = sample.elapsed_seconds
        self._peak_rss = max(self._peak_rss, sample.process_rss_mb)
        if sample.gpu_memory_mb is not None:
            self._peak_gpu = (
                sample.gpu_memory_mb if self._peak_gpu is None else max(self._peak_gpu, sample.gpu_memory_mb)
            )
        self._peak_latency = max(self._peak_latency, sample.p95_latency_ms)
        self._frames = max(self._frames, sample.frames_processed)
        self._recoverable = max(self._recoverable, sample.recoverable_errors)
        self._fatal = max(self._fatal, sample.fatal_errors)

    def report(self) -> SoakReport:
        denominator = self._samples * self._sum_xx - self._sum_x * self._sum_x
        slope = (
            0.0
            if self._samples < 2 or abs(denominator) <= 1e-12
            else (self._samples * self._sum_xy - self._sum_x * self._sum_y) / denominator
        )
        return SoakReport(
            samples=self._samples,
            duration_hours=self._duration_seconds / 3600.0,
            memory_slope_mb_per_hour=slope,
            peak_process_rss_mb=self._peak_rss,
            peak_gpu_memory_mb=self._peak_gpu,
            peak_p95_latency_ms=self._peak_latency,
            frames_processed=self._frames,
            recoverable_errors=self._recoverable,
            fatal_errors=self._fatal,
        )


class FaultScenario(str, Enum):
    NETWORK_LOSS = "network_loss"
    STREAM_LOSS = "stream_loss"
    CORRUPT_FRAME = "corrupt_frame"
    CLOCK_DRIFT = "clock_drift"
    PROCESS_RESTART = "process_restart"


@dataclass(frozen=True, slots=True)
class FaultResult:
    scenario: FaultScenario
    recovered: bool
    recovery_seconds: float | None
    evidence_reference: str

    def __post_init__(self) -> None:
        if self.recovery_seconds is not None and self.recovery_seconds < 0.0:
            raise ValueError("fault recovery_seconds must be non-negative")
        if not self.evidence_reference.strip():
            raise ValueError("fault evidence_reference must not be empty")
        if self.recovered and self.recovery_seconds is None:
            raise ValueError("recovered fault results require recovery_seconds")


def validate_fault_matrix(results: tuple[FaultResult, ...]) -> None:
    scenarios = [result.scenario for result in results]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("fault scenarios must be unique")
    missing = set(FaultScenario) - set(scenarios)
    if missing:
        raise ValueError(f"fault matrix is incomplete: {', '.join(sorted(item.value for item in missing))}")
    failed = [result.scenario.value for result in results if not result.recovered]
    if failed:
        raise ValueError(f"fault recovery failed: {', '.join(failed)}")
