"""Bounded offline trajectory interpolation for short recovered occlusions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from seahunter.schemas import InferenceMethod, ObservationKind, TrackState


class TrackStateWriter(Protocol):
    """Minimal writer contract used by the recovery fan-out."""

    def write(self, states: Sequence[TrackState]) -> None: ...

    def close(self) -> None: ...


class TrajectoryRecoveryWriter:
    """Delay a bounded window and interpolate gaps closed by observations.

    Live tracker states remain causal Kalman extrapolations. This writer is an
    output-only, bounded-latency layer: when the same identity is observed again
    within ``maximum_gap_frames``, buffered inferred boxes are replaced by a
    straight interpolation between the two observations. Longer or unclosed
    gaps stay explicitly marked as extrapolated.
    """

    def __init__(self, sinks: Sequence[TrackStateWriter], *, maximum_gap_frames: int = 3) -> None:
        if maximum_gap_frames <= 0:
            raise ValueError("maximum_gap_frames must be positive")
        if not sinks:
            raise ValueError("at least one trajectory recovery sink is required")
        self.sinks = tuple(sinks)
        self.maximum_gap_frames = maximum_gap_frames
        self._pending: dict[int, list[TrackState]] = {}
        self._last_observed: dict[int, TrackState] = {}
        self._last_frame_id: int | None = None
        self._closed = False
        self.frames_received = 0
        self.states_received = 0
        self.interpolated_gaps = 0
        self.interpolated_points = 0
        self.extrapolated_points = 0

    def write(self, states: Sequence[TrackState]) -> None:
        if self._closed:
            raise RuntimeError("trajectory recovery writer is closed")
        if not states:
            return
        frame_ids = {state.frame_id for state in states}
        if len(frame_ids) != 1:
            raise ValueError("one recovery write must contain exactly one frame")
        frame_id = next(iter(frame_ids))
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("trajectory recovery frames must increase strictly")
        track_ids = [state.track_id for state in states]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("trajectory recovery input must contain unique track IDs per frame")

        ordered = sorted(states, key=lambda state: state.track_id)
        self._pending[frame_id] = ordered
        self._last_frame_id = frame_id
        self.frames_received += 1
        self.states_received += len(ordered)
        self.extrapolated_points += sum(state.inference_method is InferenceMethod.EXTRAPOLATED for state in ordered)

        for state in ordered:
            if state.observation is not ObservationKind.OBSERVED:
                continue
            previous = self._last_observed.get(state.track_id)
            if previous is not None:
                gap = state.frame_id - previous.frame_id - 1
                if 0 < gap <= self.maximum_gap_frames:
                    replaced = self._interpolate_gap(previous, state)
                    if replaced:
                        self.interpolated_gaps += 1
                        self.interpolated_points += replaced
                        self.extrapolated_points -= replaced
            self._last_observed[state.track_id] = state

        self._flush_through(frame_id - self.maximum_gap_frames)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self._flush_through(None)
        except BaseException as exc:
            first_error = exc
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def summary(self) -> dict[str, int]:
        """Return deterministic bounded-recovery counters."""

        return {
            "maximum_gap_frames": self.maximum_gap_frames,
            "frames_received": self.frames_received,
            "states_received": self.states_received,
            "interpolated_gaps": self.interpolated_gaps,
            "interpolated_points": self.interpolated_points,
            "extrapolated_points": self.extrapolated_points,
        }

    def _interpolate_gap(self, start: TrackState, end: TrackState) -> int:
        span = end.frame_id - start.frame_id
        if span <= 1:
            return 0
        duration_seconds = (end.captured_at - start.captured_at).total_seconds()
        if duration_seconds > 0.0:
            start_center = _bbox_center(start.bbox_xyxy)
            end_center = _bbox_center(end.bbox_xyxy)
            velocity = (
                (end_center[0] - start_center[0]) / duration_seconds,
                (end_center[1] - start_center[1]) / duration_seconds,
            )
        else:
            velocity = end.velocity_xy_px_s

        replaced = 0
        for frame_id in range(start.frame_id + 1, end.frame_id):
            states = self._pending.get(frame_id)
            if states is None:
                continue
            for index, state in enumerate(states):
                if state.track_id != end.track_id or state.observation is not ObservationKind.INFERRED:
                    continue
                alpha = (frame_id - start.frame_id) / span
                states[index] = replace(
                    state,
                    bbox_xyxy=_lerp_quad(start.bbox_xyxy, end.bbox_xyxy, alpha),
                    confidence=min(state.confidence, _lerp(start.confidence, end.confidence, alpha)),
                    velocity_xy_px_s=velocity,
                    covariance_xy=state.covariance_xy,
                    inference_method=InferenceMethod.INTERPOLATED,
                )
                replaced += 1
                break
        return replaced

    def _flush_through(self, maximum_frame_id: int | None) -> None:
        selected = sorted(
            frame_id for frame_id in self._pending if maximum_frame_id is None or frame_id <= maximum_frame_id
        )
        for frame_id in selected:
            states = self._pending.pop(frame_id)
            for sink in self.sinks:
                sink.write(states)


def _lerp(start: float, end: float, alpha: float) -> float:
    return (1.0 - alpha) * start + alpha * end


def _lerp_quad(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return (
        _lerp(start[0], end[0], alpha),
        _lerp(start[1], end[1], alpha),
        _lerp(start[2], end[2], alpha),
        _lerp(start[3], end[3], alpha),
    )


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
