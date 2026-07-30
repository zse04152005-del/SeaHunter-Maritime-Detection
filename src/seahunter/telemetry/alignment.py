"""Quality-gated nearest telemetry alignment for decoded frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from seahunter.schemas import FramePacket, TelemetryPacket


@dataclass(frozen=True, slots=True)
class FrameTelemetryAlignmentConfig:
    video_latency_seconds: float = 0.0
    maximum_alignment_error_seconds: float = 0.1
    minimum_telemetry_quality: float = 0.5
    maximum_packets_per_source: int = 512

    def __post_init__(self) -> None:
        if not isfinite(self.video_latency_seconds):
            raise ValueError("video_latency_seconds must be finite")
        if self.maximum_alignment_error_seconds <= 0.0:
            raise ValueError("maximum_alignment_error_seconds must be positive")
        if not 0.0 <= self.minimum_telemetry_quality <= 1.0:
            raise ValueError("minimum_telemetry_quality must be within [0, 1]")
        if self.maximum_packets_per_source <= 0:
            raise ValueError("maximum_packets_per_source must be positive")


@dataclass(frozen=True, slots=True)
class FrameTelemetryAlignment:
    packet: TelemetryPacket | None
    accepted: bool
    alignment_error_seconds: float | None
    quality: float
    reason: str


class FrameTelemetryAligner:
    def __init__(self, config: FrameTelemetryAlignmentConfig | None = None) -> None:
        self.config = config or FrameTelemetryAlignmentConfig()
        self._history: dict[str, deque[TelemetryPacket]] = {}

    def add(self, packet: TelemetryPacket) -> None:
        history = self._history.setdefault(
            packet.source_id,
            deque(maxlen=self.config.maximum_packets_per_source),
        )
        if history and packet.captured_at <= history[-1].captured_at:
            raise ValueError("telemetry captured_at must increase strictly per source")
        history.append(packet)

    def align(self, frame: FramePacket) -> FrameTelemetryAlignment:
        history = self._history.get(frame.source_id)
        if not history:
            return FrameTelemetryAlignment(None, False, None, 0.0, "missing_telemetry")
        target = frame.captured_at - timedelta(seconds=self.config.video_latency_seconds)
        packet = min(history, key=lambda candidate: abs((candidate.captured_at - target).total_seconds()))
        error = abs((packet.captured_at - target).total_seconds())
        if packet.quality < self.config.minimum_telemetry_quality:
            return FrameTelemetryAlignment(packet, False, error, 0.0, "telemetry_quality_below_gate")
        if error > self.config.maximum_alignment_error_seconds:
            return FrameTelemetryAlignment(packet, False, error, 0.0, "alignment_error_exceeded")
        time_quality = max(0.0, 1.0 - error / self.config.maximum_alignment_error_seconds)
        return FrameTelemetryAlignment(packet, True, error, packet.quality * time_quality, "accepted")

    def reset(self, source_id: str | None = None) -> None:
        if source_id is None:
            self._history.clear()
        else:
            self._history.pop(source_id, None)
