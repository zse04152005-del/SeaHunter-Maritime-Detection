"""BoT-SORT motion baseline with robust visual global-motion compensation."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from seahunter.schemas import Detection, FramePacket, ObservationKind, TelemetryPacket, TrackLifecycle, TrackState

from .appearance import (
    AppearanceEncoder,
    AppearanceObservation,
    AppearanceQualityConfig,
    AppearanceQualityGate,
    HistogramAppearanceEncoder,
    aggregate_template,
    cosine_similarity,
)
from .bytetrack import ByteTrackConfig, ByteTracker, _bbox_iou, _Track
from .global_motion import (
    GlobalMotionEstimate,
    GlobalMotionEstimator,
    SparseOpticalFlowConfig,
    SparseOpticalFlowGMC,
)
from .motion_fusion import MotionFusionConfig, fuse_global_motion
from .telemetry_motion import TelemetryMotionEstimator


@dataclass(frozen=True, slots=True)
class BoTSORTConfig(ByteTrackConfig):
    """BoT-SORT motion-branch configuration; appearance ReID is gated separately."""

    gmc_enabled: bool = True
    global_motion: SparseOpticalFlowConfig = SparseOpticalFlowConfig()
    motion_fusion: MotionFusionConfig = MotionFusionConfig()
    reid_enabled: bool = False
    appearance_quality: AppearanceQualityConfig = AppearanceQualityConfig()
    minimum_appearance_similarity: float = 0.75
    reid_motion_gate_threshold: float = 50.0
    maximum_reid_age_frames: int = 30
    template_update_rate: float = 0.25

    def __post_init__(self) -> None:
        ByteTrackConfig.__post_init__(self)
        if not 0.0 <= self.minimum_appearance_similarity <= 1.0:
            raise ValueError("minimum_appearance_similarity must be within [0, 1]")
        if not isfinite(self.reid_motion_gate_threshold) or self.reid_motion_gate_threshold <= 0:
            raise ValueError("reid_motion_gate_threshold must be finite and positive")
        if self.maximum_reid_age_frames < 0:
            raise ValueError("maximum_reid_age_frames must be non-negative")
        if self.reid_enabled and self.maximum_reid_age_frames > self.max_lost_frames:
            raise ValueError("maximum_reid_age_frames cannot exceed max_lost_frames when ReID is enabled")
        if not 0.0 < self.template_update_rate <= 1.0:
            raise ValueError("template_update_rate must be within (0, 1]")


class BoTSORTTracker(ByteTracker):
    """ByteTrack plus camera motion and an optional quality-gated ReID layer."""

    def __init__(
        self,
        config: BoTSORTConfig | None = None,
        *,
        motion_estimator: GlobalMotionEstimator | None = None,
        motion_prior: TelemetryMotionEstimator | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
    ) -> None:
        self.botsort_config = config or BoTSORTConfig()
        super().__init__(self.botsort_config)
        self._motion_estimator: GlobalMotionEstimator | None
        if not self.botsort_config.gmc_enabled:
            self._motion_estimator = None
        elif motion_estimator is not None:
            self._motion_estimator = motion_estimator
        else:
            self._motion_estimator = SparseOpticalFlowGMC(self.botsort_config.global_motion)
        self._motion_prior = motion_prior
        self._appearance_gate: AppearanceQualityGate | None
        if self.botsort_config.reid_enabled:
            encoder = appearance_encoder or HistogramAppearanceEncoder()
            self._appearance_gate = AppearanceQualityGate(
                self.botsort_config.appearance_quality,
                encoder=encoder,
            )
            appearance_id = encoder.encoder_id
        else:
            self._appearance_gate = None
            appearance_id = "0"
        estimator_id = "disabled" if self._motion_estimator is None else self._motion_estimator.estimator_id
        prior_id = "disabled" if self._motion_prior is None else self._motion_prior.estimator_id
        self._tracker_id = self._tracker_id.replace("bytetrack:v1", "botsort:v1", 1) + (
            f":gmc={estimator_id}:prior={prior_id}:reid={appearance_id}"
        )
        self._previous_frame: FramePacket | None = None
        self._previous_detections: tuple[Detection, ...] = ()
        self._last_motion = GlobalMotionEstimate.identity("no_previous_frame", source="none")
        self._motion_frames = 0
        self._motion_applied_frames = 0
        self._motion_quality_sum = 0.0
        self._motion_source_counts: dict[str, int] = {}
        self._appearance_observations: dict[int, AppearanceObservation] = {}
        self._appearance_eligible = 0
        self._appearance_bypass_counts: dict[str, int] = {}
        self._template_updates = 0
        self._reid_pairs_evaluated = 0
        self._reid_pairs_motion_gated = 0
        self._reid_matches = 0

    @property
    def last_global_motion(self) -> GlobalMotionEstimate:
        return self._last_motion

    def motion_summary(self) -> dict[str, object]:
        """Return deterministic sequence-level GMC acceptance diagnostics."""

        return {
            "estimator_id": "disabled" if self._motion_estimator is None else self._motion_estimator.estimator_id,
            "frames_evaluated": self._motion_frames,
            "frames_applied": self._motion_applied_frames,
            "application_rate": (
                0.0 if self._motion_frames == 0 else round(self._motion_applied_frames / self._motion_frames, 6)
            ),
            "mean_applied_quality": (
                0.0
                if self._motion_applied_frames == 0
                else round(self._motion_quality_sum / self._motion_applied_frames, 6)
            ),
            "source_counts": dict(sorted(self._motion_source_counts.items())),
            "last_fallback_reason": self._last_motion.fallback_reason,
            "last_source": self._last_motion.source,
            "last_fusion_reason": self._last_motion.fusion_reason,
        }

    def push_telemetry(self, packet: TelemetryPacket) -> None:
        """Submit telemetry to the optional bounded motion-prior history."""

        if self._motion_prior is not None:
            self._motion_prior.add(packet)

    def appearance_summary(self) -> dict[str, object]:
        """Return deterministic quality-gate, template, and ReID counters."""

        return {
            "enabled": self._appearance_gate is not None,
            "eligible_observations": self._appearance_eligible,
            "bypass_counts": dict(sorted(self._appearance_bypass_counts.items())),
            "template_updates": self._template_updates,
            "reid_pairs_evaluated": self._reid_pairs_evaluated,
            "reid_pairs_motion_gated": self._reid_pairs_motion_gated,
            "reid_matches": self._reid_matches,
        }

    def update(self, frame: FramePacket, detections: Sequence[Detection]) -> list[TrackState]:
        self._appearance_observations = self._observe_appearances(frame, detections)
        try:
            states = super().update(frame, detections)
            self._previous_frame = frame
            self._previous_detections = tuple(detections)
            return states
        finally:
            self._appearance_observations = {}

    def reset(self) -> None:
        super().reset()
        if self._motion_estimator is not None:
            self._motion_estimator.reset()
        if self._motion_prior is not None:
            self._motion_prior.reset()
        self._previous_frame = None
        self._previous_detections = ()
        self._last_motion = GlobalMotionEstimate.identity("no_previous_frame", source="none")
        self._motion_frames = 0
        self._motion_applied_frames = 0
        self._motion_quality_sum = 0.0
        self._motion_source_counts.clear()
        self._appearance_observations = {}
        self._appearance_eligible = 0
        self._appearance_bypass_counts.clear()
        self._template_updates = 0
        self._reid_pairs_evaluated = 0
        self._reid_pairs_motion_gated = 0
        self._reid_matches = 0

    def _activate_track(self, detection: Detection) -> _Track:
        track = super()._activate_track(detection)
        self._apply_appearance_observation(track, detection)
        return track

    def _update_track(
        self,
        track: _Track,
        detection: Detection,
        score: float,
        *,
        stage: str,
        motion_distance: float | None,
    ) -> None:
        super()._update_track(
            track,
            detection,
            score,
            stage=stage,
            motion_distance=motion_distance,
        )
        self._apply_appearance_observation(
            track,
            detection,
            update_template=stage != "low",
        )

    def _associate_remaining(
        self,
        frame: FramePacket,
        track_indices: Sequence[int],
        high_detections: Sequence[Detection],
        unmatched_high: Sequence[int],
        observed_indices: set[int],
    ) -> tuple[list[int], list[int]]:
        del frame
        if self._appearance_gate is None or not track_indices or not unmatched_high:
            return list(track_indices), list(unmatched_high)

        candidate_tracks = [
            index
            for index in track_indices
            if self._tracks[index].lifecycle in {TrackLifecycle.CONFIRMED, TrackLifecycle.LOST}
            and self._tracks[index].appearance_template is not None
            and self._tracks[index].time_since_update <= self.botsort_config.maximum_reid_age_frames
        ]
        candidate_detections = [
            index for index in unmatched_high if self._appearance_observations[id(high_detections[index])].eligible
        ]
        if not candidate_tracks or not candidate_detections:
            return list(track_indices), list(unmatched_high)

        np: Any = importlib.import_module("numpy")
        optimize: Any = importlib.import_module("scipy.optimize")
        invalid_cost = 1e6
        costs = np.full((len(candidate_tracks), len(candidate_detections)), invalid_cost, dtype=float)
        similarities = np.full_like(costs, -1.0)
        motion_distances = np.full_like(costs, np.nan)
        for row, track_index in enumerate(candidate_tracks):
            track = self._tracks[track_index]
            assert track.appearance_template is not None
            for column, detection_index in enumerate(candidate_detections):
                detection = high_detections[detection_index]
                if self.config.class_aware and track.class_id != detection.class_id:
                    continue
                self._reid_pairs_evaluated += 1
                motion_distance = self._kalman.gating_distance(track.mean, track.covariance, detection.bbox_xyxy)
                motion_distances[row, column] = motion_distance
                if motion_distance > self.botsort_config.reid_motion_gate_threshold:
                    self._reid_pairs_motion_gated += 1
                    continue
                observation = self._appearance_observations[id(detection)]
                assert observation.embedding is not None
                similarity = cosine_similarity(track.appearance_template, observation.embedding)
                similarities[row, column] = similarity
                if similarity >= self.botsort_config.minimum_appearance_similarity:
                    costs[row, column] = 1.0 - similarity

        rows, columns = optimize.linear_sum_assignment(costs)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if costs[row, column] >= invalid_cost:
                continue
            track_index = candidate_tracks[row]
            detection_index = candidate_detections[column]
            detection = high_detections[detection_index]
            motion_distance = float(motion_distances[row, column])
            similarity = float(similarities[row, column])
            iou = _bbox_iou(self._kalman.to_xyxy(self._tracks[track_index].mean), detection.bbox_xyxy)
            self._update_track(
                self._tracks[track_index],
                detection,
                iou,
                stage="reid",
                motion_distance=motion_distance,
            )
            self._tracks[track_index].appearance_score = similarity
            observed_indices.add(track_index)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            self._reid_matches += 1
        return (
            [index for index in track_indices if index not in matched_tracks],
            [index for index in unmatched_high if index not in matched_detections],
        )

    def _observe_appearances(
        self,
        frame: FramePacket,
        detections: Sequence[Detection],
    ) -> dict[int, AppearanceObservation]:
        if self._appearance_gate is None:
            return {}
        observations: dict[int, AppearanceObservation] = {}
        for detection in detections:
            observation = self._appearance_gate.observe(frame, detection, detections)
            observations[id(detection)] = observation
            if observation.eligible:
                self._appearance_eligible += 1
            else:
                reason = observation.bypass_reason or "unknown"
                self._appearance_bypass_counts[reason] = self._appearance_bypass_counts.get(reason, 0) + 1
        return observations

    def _apply_appearance_observation(
        self,
        track: _Track,
        detection: Detection,
        *,
        update_template: bool = True,
    ) -> None:
        if self._appearance_gate is None:
            track.reid_eligible = False
            track.reid_bypass_reason = "reid_disabled"
            return
        observation = self._appearance_observations[id(detection)]
        track.reid_eligible = observation.eligible
        track.reid_bypass_reason = observation.bypass_reason
        if not update_template or not observation.eligible or observation.embedding is None:
            return
        track.appearance_template = aggregate_template(
            track.appearance_template,
            observation.embedding,
            quality=observation.quality,
            update_rate=self.botsort_config.template_update_rate,
        )
        track.appearance_samples += 1
        track.appearance_quality = observation.quality
        self._template_updates += 1

    def _after_prediction(self, frame: FramePacket, detections: Sequence[Detection]) -> None:
        if self._previous_frame is None:
            self._last_motion = GlobalMotionEstimate.identity("no_previous_frame", source="none")
            return

        if not self.botsort_config.gmc_enabled or self._motion_estimator is None:
            visual = GlobalMotionEstimate.identity("gmc_disabled")
        else:
            visual = self._motion_estimator.estimate(
                self._previous_frame,
                frame,
                previous_detections=self._previous_detections,
                current_detections=detections,
            )
        if self._motion_prior is None:
            self._last_motion = replace(visual, visual_quality=visual.quality)
        else:
            prior = self._motion_prior.estimate(
                self._previous_frame,
                frame,
                previous_detections=self._previous_detections,
                current_detections=detections,
            )
            self._last_motion = fuse_global_motion(visual, prior, frame, self.botsort_config.motion_fusion)
        self._motion_frames += 1
        if not self._last_motion.applied:
            return
        self._motion_applied_frames += 1
        self._motion_quality_sum += self._last_motion.quality
        self._motion_source_counts[self._last_motion.source] = (
            self._motion_source_counts.get(self._last_motion.source, 0) + 1
        )
        for track in self._tracks:
            if track.lifecycle is TrackLifecycle.REMOVED:
                continue
            track.mean, track.covariance = self._kalman.apply_affine(
                track.mean,
                track.covariance,
                self._last_motion.affine_2x3,
            )

    def _to_state(
        self,
        track: _Track,
        frame: FramePacket,
        observation: ObservationKind,
    ) -> TrackState:
        state = super()._to_state(track, frame, observation)
        return replace(
            state,
            global_motion_affine=self._last_motion.affine_2x3,
            global_motion_quality=self._last_motion.quality,
            global_motion_applied=self._last_motion.applied,
            global_motion_fallback_reason=self._last_motion.fallback_reason,
            global_motion_source=self._last_motion.source,
            global_motion_visual_quality=self._last_motion.visual_quality,
            global_motion_prior_quality=self._last_motion.prior_quality,
            global_motion_fusion_reason=self._last_motion.fusion_reason,
        )
