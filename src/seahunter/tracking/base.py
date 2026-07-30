"""Framework-neutral multi-object tracker interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from seahunter.schemas import Detection, FramePacket, TelemetryPacket, TrackState


class MultiObjectTracker(Protocol):
    """Minimal stateful tracker contract used by the video pipeline."""

    @property
    def tracker_id(self) -> str:
        """Return a stable algorithm/configuration identifier."""

    def update(self, frame: FramePacket, detections: Sequence[Detection]) -> Sequence[TrackState]:
        """Advance the tracker by one frame and return emitted track states."""

    def reset(self) -> None:
        """Clear all active identities and sequence state."""


@runtime_checkable
class TelemetryAwareTracker(Protocol):
    """Optional tracker capability for timestamped IMU/gimbal packets."""

    def push_telemetry(self, packet: TelemetryPacket) -> None:
        """Submit one packet before advancing its aligned video frame."""
