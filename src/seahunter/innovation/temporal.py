"""Finite-state temporal small-target monitoring on stabilized features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class TemporalSignalClass(str, Enum):
    STABLE_BACKGROUND = "stable_background"
    CAMERA_MOTION = "camera_motion"
    PERIODIC_WAVE = "periodic_wave"
    TARGET_CANDIDATE = "target_candidate"


@dataclass(frozen=True, slots=True)
class TemporalFeatureFrame:
    source_id: str
    scene_id: str
    frame_index: int
    spatial_features: tuple[float, ...]
    motion_residual: tuple[float, ...]
    wave_periodicity: tuple[float, ...]
    camera_motion: float

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.scene_id.strip() or self.frame_index < 0:
            raise ValueError("temporal frame identity is invalid")
        if not self.spatial_features or not (
            len(self.spatial_features) == len(self.motion_residual) == len(self.wave_periodicity)
        ):
            raise ValueError("temporal feature vectors must have equal non-zero length")
        values = self.spatial_features + self.motion_residual + self.wave_periodicity + (self.camera_motion,)
        if not all(isfinite(value) for value in values):
            raise ValueError("temporal feature values must be finite")
        if not 0.0 <= self.camera_motion <= 1.0 or any(not 0.0 <= value <= 1.0 for value in self.wave_periodicity):
            raise ValueError("camera motion and wave periodicity must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalMonitorConfig:
    history_frames: int = 4
    maximum_frame_gap: int = 3
    spatial_weight: float = 0.35
    temporal_weight: float = 0.35
    motion_weight: float = 0.3
    camera_motion_gate: float = 0.75
    wave_gate: float = 0.7
    candidate_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.history_frames <= 0 or self.maximum_frame_gap <= 0:
            raise ValueError("temporal history and gap controls must be positive")
        weights = (self.spatial_weight, self.temporal_weight, self.motion_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("temporal fusion weights must be non-negative with positive sum")
        gates = (self.camera_motion_gate, self.wave_gate, self.candidate_threshold)
        if any(not 0.0 <= gate <= 1.0 for gate in gates):
            raise ValueError("temporal gates must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalMonitorOutput:
    frame_index: int
    fused_scores: tuple[float, ...]
    signal_classes: tuple[TemporalSignalClass, ...]
    history_size: int
    reset_reason: str | None


@dataclass(slots=True)
class _TemporalState:
    scene_id: str
    last_frame_index: int
    history: deque[tuple[float, ...]]


class TemporalSmallTargetMonitor:
    def __init__(self, config: TemporalMonitorConfig | None = None) -> None:
        self.config = config or TemporalMonitorConfig()
        self._states: dict[str, _TemporalState] = {}

    def update(self, frame: TemporalFeatureFrame) -> TemporalMonitorOutput:
        state = self._states.get(frame.source_id)
        reset_reason: str | None = None
        if state is not None and frame.frame_index <= state.last_frame_index:
            raise ValueError("temporal frame indexes must increase strictly per source")
        if state is None:
            reset_reason = "new_source"
        elif state.scene_id != frame.scene_id:
            reset_reason = "scene_change"
        elif frame.frame_index - state.last_frame_index > self.config.maximum_frame_gap:
            reset_reason = "stream_gap"
        elif state.history and len(state.history[0]) != len(frame.spatial_features):
            reset_reason = "feature_shape_change"
        if reset_reason is not None:
            state = _TemporalState(frame.scene_id, frame.frame_index, deque(maxlen=self.config.history_frames))
            self._states[frame.source_id] = state

        assert state is not None
        history_mean = _mean_vectors(tuple(state.history), len(frame.spatial_features))
        total_weight = self.config.spatial_weight + self.config.temporal_weight + self.config.motion_weight
        scores: list[float] = []
        classes: list[TemporalSignalClass] = []
        for index, spatial in enumerate(frame.spatial_features):
            temporal_change = abs(spatial - history_mean[index]) if state.history else 0.0
            raw = (
                self.config.spatial_weight * _unit(spatial)
                + self.config.temporal_weight * _unit(temporal_change)
                + self.config.motion_weight * _unit(abs(frame.motion_residual[index]))
            ) / total_weight
            score = max(0.0, min(1.0, raw * (1.0 - 0.5 * frame.camera_motion)))
            if frame.camera_motion >= self.config.camera_motion_gate:
                signal_class = TemporalSignalClass.CAMERA_MOTION
            elif frame.wave_periodicity[index] >= self.config.wave_gate:
                signal_class = TemporalSignalClass.PERIODIC_WAVE
                score *= 1.0 - frame.wave_periodicity[index]
            elif score >= self.config.candidate_threshold:
                signal_class = TemporalSignalClass.TARGET_CANDIDATE
            else:
                signal_class = TemporalSignalClass.STABLE_BACKGROUND
            scores.append(score)
            classes.append(signal_class)
        state.history.append(frame.spatial_features)
        state.last_frame_index = frame.frame_index
        state.scene_id = frame.scene_id
        return TemporalMonitorOutput(frame.frame_index, tuple(scores), tuple(classes), len(state.history), reset_reason)

    def reset(self, source_id: str | None = None) -> None:
        if source_id is None:
            self._states.clear()
        else:
            self._states.pop(source_id, None)


def _mean_vectors(history: tuple[tuple[float, ...], ...], length: int) -> tuple[float, ...]:
    if not history:
        return (0.0,) * length
    return tuple(sum(vector[index] for vector in history) / len(history) for index in range(length))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))
