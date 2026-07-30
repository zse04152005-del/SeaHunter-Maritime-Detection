"""Bounded affine clock synchronization with explicit quality gates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class ClockSyncConfig:
    minimum_samples: int = 5
    maximum_samples: int = 256
    maximum_absolute_drift_ppm: float = 2_000.0
    maximum_rmse_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        if self.maximum_samples < self.minimum_samples:
            raise ValueError("maximum_samples cannot be below minimum_samples")
        if self.maximum_absolute_drift_ppm <= 0.0 or self.maximum_rmse_ms <= 0.0:
            raise ValueError("clock drift and residual gates must be positive")


@dataclass(frozen=True, slots=True)
class ClockSyncEstimate:
    accepted: bool
    sample_count: int
    offset_seconds: float | None
    scale: float | None
    drift_ppm: float | None
    rmse_ms: float | None
    quality: float
    reason: str
    reset_count: int


class ClockSynchronizer:
    """Fit UTC seconds = offset + scale * device seconds in a bounded window."""

    def __init__(self, config: ClockSyncConfig | None = None) -> None:
        self.config = config or ClockSyncConfig()
        self._samples: deque[tuple[float, float]] = deque(maxlen=self.config.maximum_samples)
        self._last_source_seconds: float | None = None
        self._reset_count = 0

    def add_sample(self, source_seconds: float, reference_at: datetime) -> ClockSyncEstimate:
        if not isfinite(source_seconds) or source_seconds < 0.0:
            raise ValueError("source_seconds must be finite and non-negative")
        if reference_at.tzinfo is None or reference_at.utcoffset() is None:
            raise ValueError("reference_at must be timezone-aware")
        if self._last_source_seconds is not None and source_seconds == self._last_source_seconds:
            return self.estimate()
        if self._last_source_seconds is not None and source_seconds < self._last_source_seconds:
            self._samples.clear()
            self._reset_count += 1
        self._samples.append((source_seconds, reference_at.astimezone(timezone.utc).timestamp()))
        self._last_source_seconds = source_seconds
        return self.estimate()

    def estimate(self) -> ClockSyncEstimate:
        sample_count = len(self._samples)
        if sample_count < self.config.minimum_samples:
            return ClockSyncEstimate(
                accepted=False,
                sample_count=sample_count,
                offset_seconds=None,
                scale=None,
                drift_ppm=None,
                rmse_ms=None,
                quality=0.0,
                reason="insufficient_samples",
                reset_count=self._reset_count,
            )
        source_mean = sum(sample[0] for sample in self._samples) / sample_count
        reference_mean = sum(sample[1] for sample in self._samples) / sample_count
        denominator = sum((source - source_mean) ** 2 for source, _ in self._samples)
        if denominator <= 1e-12:
            return ClockSyncEstimate(
                accepted=False,
                sample_count=sample_count,
                offset_seconds=None,
                scale=None,
                drift_ppm=None,
                rmse_ms=None,
                quality=0.0,
                reason="degenerate_source_clock",
                reset_count=self._reset_count,
            )
        scale = (
            sum((source - source_mean) * (reference - reference_mean) for source, reference in self._samples)
            / denominator
        )
        offset = reference_mean - scale * source_mean
        residuals = [reference - (offset + scale * source) for source, reference in self._samples]
        rmse_ms = sqrt(sum(residual * residual for residual in residuals) / sample_count) * 1_000.0
        drift_ppm = (scale - 1.0) * 1_000_000.0
        drift_quality = max(0.0, 1.0 - abs(drift_ppm) / self.config.maximum_absolute_drift_ppm)
        residual_quality = max(0.0, 1.0 - rmse_ms / self.config.maximum_rmse_ms)
        sample_quality = min(1.0, sample_count / (2.0 * self.config.minimum_samples))
        quality = drift_quality * residual_quality * sample_quality
        if abs(drift_ppm) > self.config.maximum_absolute_drift_ppm:
            accepted, reason = False, "drift_gate_exceeded"
        elif rmse_ms > self.config.maximum_rmse_ms:
            accepted, reason = False, "residual_gate_exceeded"
        else:
            accepted, reason = True, "accepted"
        return ClockSyncEstimate(
            accepted=accepted,
            sample_count=sample_count,
            offset_seconds=offset,
            scale=scale,
            drift_ppm=drift_ppm,
            rmse_ms=rmse_ms,
            quality=quality,
            reason=reason,
            reset_count=self._reset_count,
        )

    def align(self, source_seconds: float) -> datetime | None:
        if not isfinite(source_seconds) or source_seconds < 0.0:
            raise ValueError("source_seconds must be finite and non-negative")
        estimate = self.estimate()
        if not estimate.accepted or estimate.offset_seconds is None or estimate.scale is None:
            return None
        return datetime.fromtimestamp(
            estimate.offset_seconds + estimate.scale * source_seconds,
            tz=timezone.utc,
        )

    def reset(self) -> None:
        self._samples.clear()
        self._last_source_seconds = None
        self._reset_count += 1
