"""Adapter from per-frame detector results to stateful tracking outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from seahunter.schemas import TrackState
from seahunter.tracking import MultiObjectTracker

from .results import FrameResult


class TrackStateSink(Protocol):
    """Streaming consumer for normalized track states."""

    def write(self, states: Sequence[TrackState]) -> None: ...

    def close(self) -> None: ...


class TrackingSink:
    """Run a tracker as a detector-result sink and fan out track states."""

    def __init__(self, tracker: MultiObjectTracker, *, sinks: Sequence[TrackStateSink] = ()) -> None:
        self.tracker = tracker
        self.sinks = tuple(sinks)
        self.latest_states: tuple[TrackState, ...] = ()
        self.frames_processed = 0
        self.states_emitted = 0
        self._closed = False

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("tracking sink is closed")
        states = tuple(self.tracker.update(result.frame, result.detections))
        self.latest_states = states
        for sink in self.sinks:
            sink.write(states)
        self.frames_processed += 1
        self.states_emitted += len(states)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
