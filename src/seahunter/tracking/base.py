"""Framework-neutral multi-object tracker interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from seahunter.schemas import Detection, FramePacket, TrackState


class MultiObjectTracker(Protocol):
    """Minimal stateful tracker contract used by the video pipeline."""

    @property
    def tracker_id(self) -> str:
        """Return a stable algorithm/configuration identifier."""

    def update(self, frame: FramePacket, detections: Sequence[Detection]) -> Sequence[TrackState]:
        """Advance the tracker by one frame and return emitted track states."""

    def reset(self) -> None:
        """Clear all active identities and sequence state."""
