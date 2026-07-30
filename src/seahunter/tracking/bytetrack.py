"""Framework-neutral ByteTrack baseline with two-stage IoU association."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from seahunter.schemas import (
    Detection,
    FramePacket,
    ObservationKind,
    TrackLifecycle,
    TrackLossReason,
    TrackState,
)

from .kalman import KalmanXYWH


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    """Thresholds and lifecycle controls for the ByteTrack baseline."""

    high_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.1
    new_track_threshold: float = 0.7
    first_match_iou_threshold: float = 0.3
    second_match_iou_threshold: float = 0.2
    max_lost_frames: int = 30
    minimum_confirmed_hits: int = 1
    frame_rate: float = 30.0
    inferred_confidence_decay: float = 0.9
    class_aware: bool = True
    emit_lost_predictions: bool = False
    motion_gating_enabled: bool = True
    motion_gate_threshold: float = 13.276704

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= self.high_confidence_threshold <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
        if not self.high_confidence_threshold <= self.new_track_threshold <= 1.0:
            raise ValueError("new_track_threshold must be within [high_confidence_threshold, 1]")
        if not 0.0 <= self.first_match_iou_threshold <= 1.0:
            raise ValueError("first_match_iou_threshold must be within [0, 1]")
        if not 0.0 <= self.second_match_iou_threshold <= 1.0:
            raise ValueError("second_match_iou_threshold must be within [0, 1]")
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must be non-negative")
        if self.minimum_confirmed_hits <= 0:
            raise ValueError("minimum_confirmed_hits must be positive")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if not 0.0 <= self.inferred_confidence_decay <= 1.0:
            raise ValueError("inferred_confidence_decay must be within [0, 1]")
        if not isfinite(self.motion_gate_threshold) or self.motion_gate_threshold <= 0:
            raise ValueError("motion_gate_threshold must be finite and positive")


@dataclass(slots=True)
class _Track:
    track_id: int
    mean: Any
    covariance: Any
    class_id: int
    confidence: float
    lifecycle: TrackLifecycle
    age_frames: int
    hits: int
    time_since_update: int
    association_score: float | None
    association_stage: str | None
    motion_gate_distance: float | None
    appearance_score: float | None
    reid_eligible: bool | None
    reid_bypass_reason: str | None
    appearance_template: tuple[float, ...] | None
    appearance_samples: int
    appearance_quality: float | None
    lost_reason: TrackLossReason | None


@dataclass(frozen=True, slots=True)
class _AssociationMatch:
    track_index: int
    detection_index: int
    iou: float
    motion_distance: float | None


class ByteTracker:
    """Two-stage ByteTrack association independent of detector framework types."""

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self.config = config or ByteTrackConfig()
        self._kalman = KalmanXYWH()
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self._last_frame_id: int | None = None
        self._source_id: str | None = None
        self._tracker_id = (
            "bytetrack:v1"
            f":high={self.config.high_confidence_threshold:g}"
            f":low={self.config.low_confidence_threshold:g}"
            f":new={self.config.new_track_threshold:g}"
            f":iou={self.config.first_match_iou_threshold:g}/{self.config.second_match_iou_threshold:g}"
            f":lost={self.config.max_lost_frames}"
            f":hits={self.config.minimum_confirmed_hits}"
            f":fps={self.config.frame_rate:g}"
            f":decay={self.config.inferred_confidence_decay:g}"
            f":class={int(self.config.class_aware)}"
            f":emit_lost={int(self.config.emit_lost_predictions)}"
            f":motion_gate={int(self.config.motion_gating_enabled)}/{self.config.motion_gate_threshold:g}"
        )
        self._association_stage_counts: dict[str, int] = {}
        self._motion_pairs_evaluated = 0
        self._motion_pairs_gated = 0

    @property
    def tracker_id(self) -> str:
        return self._tracker_id

    def update(self, frame: FramePacket, detections: Sequence[Detection]) -> list[TrackState]:
        delta_frames = self._validate_frame(frame)
        for track in self._tracks:
            if track.lifecycle is TrackLifecycle.REMOVED:
                continue
            track.mean, track.covariance = self._kalman.predict(
                track.mean,
                track.covariance,
                delta_frames=delta_frames,
            )
            track.age_frames += delta_frames
            track.time_since_update += delta_frames
            track.association_score = None
            track.association_stage = None
            track.motion_gate_distance = None
            track.appearance_score = None
            track.reid_eligible = None
            track.reid_bypass_reason = None

        self._after_prediction(frame, detections)

        high_detections = [
            detection for detection in detections if detection.confidence >= self.config.high_confidence_threshold
        ]
        low_detections = [
            detection
            for detection in detections
            if self.config.low_confidence_threshold <= detection.confidence < self.config.high_confidence_threshold
        ]
        candidate_indices = [
            index for index, track in enumerate(self._tracks) if track.lifecycle is not TrackLifecycle.REMOVED
        ]
        first_matches, unmatched_tracks, unmatched_high = self._associate(
            candidate_indices,
            high_detections,
            self.config.first_match_iou_threshold,
        )

        observed_indices: set[int] = set()
        for match in first_matches:
            self._update_track(
                self._tracks[match.track_index],
                high_detections[match.detection_index],
                match.iou,
                stage="high",
                motion_distance=match.motion_distance,
            )
            observed_indices.add(match.track_index)

        second_candidates = [
            index for index in unmatched_tracks if self._tracks[index].lifecycle is TrackLifecycle.CONFIRMED
        ]
        second_matches, unmatched_second, _ = self._associate(
            second_candidates,
            low_detections,
            self.config.second_match_iou_threshold,
        )
        for match in second_matches:
            self._update_track(
                self._tracks[match.track_index],
                low_detections[match.detection_index],
                match.iou,
                stage="low",
                motion_distance=match.motion_distance,
            )
            observed_indices.add(match.track_index)

        second_candidate_set = set(second_candidates)
        remaining_unmatched = [index for index in unmatched_tracks if index not in second_candidate_set]
        remaining_unmatched.extend(unmatched_second)
        remaining_unmatched, unmatched_high = self._associate_remaining(
            frame,
            sorted(set(remaining_unmatched)),
            high_detections,
            unmatched_high,
            observed_indices,
        )
        for track_index in sorted(set(remaining_unmatched)):
            track = self._tracks[track_index]
            if track.lifecycle is TrackLifecycle.TENTATIVE:
                track.lifecycle = TrackLifecycle.REMOVED
                track.lost_reason = TrackLossReason.UNMATCHED
            else:
                track.lifecycle = TrackLifecycle.LOST
                track.lost_reason = TrackLossReason.UNMATCHED

        for detection_index in unmatched_high:
            detection = high_detections[detection_index]
            if detection.confidence < self.config.new_track_threshold:
                continue
            track = self._activate_track(detection)
            self._tracks.append(track)
            self._association_stage_counts["new"] = self._association_stage_counts.get("new", 0) + 1
            observed_indices.add(len(self._tracks) - 1)

        for track in self._tracks:
            if track.lifecycle is TrackLifecycle.LOST and track.time_since_update > self.config.max_lost_frames:
                track.lifecycle = TrackLifecycle.REMOVED
                track.lost_reason = TrackLossReason.EXPIRED

        states: list[TrackState] = []
        for index, track in enumerate(self._tracks):
            if index in observed_indices and track.lifecycle is TrackLifecycle.CONFIRMED:
                states.append(self._to_state(track, frame, ObservationKind.OBSERVED))
            elif self.config.emit_lost_predictions and track.lifecycle is TrackLifecycle.LOST:
                states.append(self._to_state(track, frame, ObservationKind.INFERRED))

        self._tracks = [track for track in self._tracks if track.lifecycle is not TrackLifecycle.REMOVED]
        self._last_frame_id = frame.frame_id
        self._source_id = frame.source_id
        return sorted(states, key=lambda state: state.track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_id = None
        self._source_id = None
        self._association_stage_counts.clear()
        self._motion_pairs_evaluated = 0
        self._motion_pairs_gated = 0

    def association_summary(self) -> dict[str, object]:
        """Return deterministic hierarchical-association diagnostics."""

        return {
            "stage_matches": dict(sorted(self._association_stage_counts.items())),
            "motion_pairs_evaluated": self._motion_pairs_evaluated,
            "motion_pairs_gated": self._motion_pairs_gated,
            "motion_gate_rate": (
                0.0
                if self._motion_pairs_evaluated == 0
                else round(self._motion_pairs_gated / self._motion_pairs_evaluated, 6)
            ),
        }

    def _validate_frame(self, frame: FramePacket) -> int:
        if self._source_id is not None and frame.source_id != self._source_id:
            raise ValueError("ByteTracker handles one source_id; call reset() before changing sources")
        if self._last_frame_id is None:
            return 1
        if frame.frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase strictly")
        return frame.frame_id - self._last_frame_id

    def _activate_track(self, detection: Detection) -> _Track:
        mean, covariance = self._kalman.initiate(detection.bbox_xyxy)
        hits = 1
        lifecycle = TrackLifecycle.CONFIRMED if hits >= self.config.minimum_confirmed_hits else TrackLifecycle.TENTATIVE
        track = _Track(
            track_id=self._next_track_id,
            mean=mean,
            covariance=covariance,
            class_id=detection.class_id,
            confidence=detection.confidence,
            lifecycle=lifecycle,
            age_frames=1,
            hits=hits,
            time_since_update=0,
            association_score=1.0,
            association_stage="new",
            motion_gate_distance=None,
            appearance_score=None,
            reid_eligible=None,
            reid_bypass_reason=None,
            appearance_template=None,
            appearance_samples=0,
            appearance_quality=None,
            lost_reason=None,
        )
        self._next_track_id += 1
        return track

    def _after_prediction(self, frame: FramePacket, detections: Sequence[Detection]) -> None:
        """Allow tracker variants to transform predicted states before association."""

        del frame, detections

    def _associate_remaining(
        self,
        frame: FramePacket,
        track_indices: Sequence[int],
        high_detections: Sequence[Detection],
        unmatched_high: Sequence[int],
        observed_indices: set[int],
    ) -> tuple[list[int], list[int]]:
        """Variant hook for a third association layer such as quality-gated ReID."""

        del frame, high_detections, observed_indices
        return list(track_indices), list(unmatched_high)

    def _update_track(
        self,
        track: _Track,
        detection: Detection,
        score: float,
        *,
        stage: str,
        motion_distance: float | None,
    ) -> None:
        track.mean, track.covariance = self._kalman.update(
            track.mean,
            track.covariance,
            detection.bbox_xyxy,
        )
        track.class_id = detection.class_id
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.association_score = score
        track.association_stage = stage
        track.motion_gate_distance = motion_distance
        self._association_stage_counts[stage] = self._association_stage_counts.get(stage, 0) + 1
        track.lost_reason = None
        if track.hits >= self.config.minimum_confirmed_hits:
            track.lifecycle = TrackLifecycle.CONFIRMED

    def _associate(
        self,
        track_indices: Sequence[int],
        detections: Sequence[Detection],
        minimum_iou: float,
    ) -> tuple[list[_AssociationMatch], list[int], list[int]]:
        if not track_indices or not detections:
            return [], list(track_indices), list(range(len(detections)))

        np: Any = importlib.import_module("numpy")
        try:
            optimize: Any = importlib.import_module("scipy.optimize")
        except ModuleNotFoundError as exc:
            raise RuntimeError("ByteTrack requires the 'tracking' project extra") from exc

        invalid_cost = 1e6
        costs = np.full((len(track_indices), len(detections)), invalid_cost, dtype=float)
        scores = np.full((len(track_indices), len(detections)), -1.0, dtype=float)
        motion_distances = np.full((len(track_indices), len(detections)), np.nan, dtype=float)
        for row, track_index in enumerate(track_indices):
            track = self._tracks[track_index]
            track_bbox = self._kalman.to_xyxy(track.mean)
            for column, detection in enumerate(detections):
                if self.config.class_aware and track.class_id != detection.class_id:
                    continue
                motion_distance: float | None = None
                if self.config.motion_gating_enabled:
                    self._motion_pairs_evaluated += 1
                    motion_distance = self._kalman.gating_distance(track.mean, track.covariance, detection.bbox_xyxy)
                    motion_distances[row, column] = motion_distance
                    if motion_distance > self.config.motion_gate_threshold:
                        self._motion_pairs_gated += 1
                        continue
                score = _bbox_iou(track_bbox, detection.bbox_xyxy)
                scores[row, column] = score
                if score >= minimum_iou:
                    costs[row, column] = 1.0 - score

        row_indices, column_indices = optimize.linear_sum_assignment(costs)
        matches: list[_AssociationMatch] = []
        matched_rows: set[int] = set()
        matched_columns: set[int] = set()
        for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
            score = float(scores[row, column])
            if costs[row, column] >= invalid_cost:
                continue
            raw_motion_distance = float(motion_distances[row, column])
            motion_distance = raw_motion_distance if np.isfinite(raw_motion_distance) else None
            matches.append(
                _AssociationMatch(
                    track_index=track_indices[row],
                    detection_index=column,
                    iou=score,
                    motion_distance=motion_distance,
                )
            )
            matched_rows.add(row)
            matched_columns.add(column)

        unmatched_tracks = [track_indices[row] for row in range(len(track_indices)) if row not in matched_rows]
        unmatched_detections = [column for column in range(len(detections)) if column not in matched_columns]
        return matches, unmatched_tracks, unmatched_detections

    def _to_state(
        self,
        track: _Track,
        frame: FramePacket,
        observation: ObservationKind,
    ) -> TrackState:
        confidence = track.confidence
        if observation is ObservationKind.INFERRED:
            confidence *= self.config.inferred_confidence_decay**track.time_since_update
        velocity = (
            float(track.mean[4]) * self.config.frame_rate,
            float(track.mean[5]) * self.config.frame_rate,
        )
        return TrackState(
            track_id=track.track_id,
            frame_id=frame.frame_id,
            captured_at=frame.captured_at,
            bbox_xyxy=self._kalman.to_xyxy(track.mean),
            class_id=track.class_id,
            confidence=max(0.0, min(1.0, confidence)),
            observation=observation,
            velocity_xy_px_s=velocity,
            covariance_xy=self._kalman.covariance_xy(track.covariance),
            lifecycle=track.lifecycle,
            tracker_id=self.tracker_id,
            age_frames=track.age_frames,
            time_since_update=track.time_since_update,
            association_score=track.association_score,
            association_stage=track.association_stage,
            motion_gate_distance=track.motion_gate_distance,
            appearance_score=track.appearance_score,
            reid_eligible=track.reid_eligible,
            reid_bypass_reason=track.reid_bypass_reason,
            lost_reason=track.lost_reason,
        )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union
