"""BoT-SORT motion baseline with robust visual global-motion compensation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from seahunter.schemas import Detection, FramePacket, ObservationKind, TelemetryPacket, TrackLifecycle, TrackState

from .bytetrack import ByteTrackConfig, ByteTracker, _Track
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

    def __post_init__(self) -> None:
        ByteTrackConfig.__post_init__(self)


class BoTSORTTracker(ByteTracker):
    """ByteTrack lifecycle plus BoT-SORT camera-motion compensation.

    This increment intentionally implements the motion branch only. Appearance
    embeddings remain disabled until maritime small-target ReID quality gates are
    available and independently validated.
    """

    def __init__(
        self,
        config: BoTSORTConfig | None = None,
        *,
        motion_estimator: GlobalMotionEstimator | None = None,
        motion_prior: TelemetryMotionEstimator | None = None,
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
        estimator_id = "disabled" if self._motion_estimator is None else self._motion_estimator.estimator_id
        prior_id = "disabled" if self._motion_prior is None else self._motion_prior.estimator_id
        self._tracker_id = (
            self._tracker_id.replace("bytetrack:v1", "botsort:v1", 1) + f":gmc={estimator_id}:prior={prior_id}:reid=0"
        )
        self._previous_frame: FramePacket | None = None
        self._previous_detections: tuple[Detection, ...] = ()
        self._last_motion = GlobalMotionEstimate.identity("no_previous_frame", source="none")
        self._motion_frames = 0
        self._motion_applied_frames = 0
        self._motion_quality_sum = 0.0
        self._motion_source_counts: dict[str, int] = {}

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

    def update(self, frame: FramePacket, detections: Sequence[Detection]) -> list[TrackState]:
        states = super().update(frame, detections)
        self._previous_frame = frame
        self._previous_detections = tuple(detections)
        return states

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
