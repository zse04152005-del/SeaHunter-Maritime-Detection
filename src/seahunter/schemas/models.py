"""Framework-neutral dataclasses for video perception and risk events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any


class SourceKind(str, Enum):
    """Supported video source families."""

    FILE = "file"
    DEVICE = "device"
    RTSP = "rtsp"
    SRT = "srt"
    HTTP = "http"


class ObservationKind(str, Enum):
    """Whether a track point came from a detector or a motion model."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class TrackLifecycle(str, Enum):
    """Lifecycle state of one tracker identity."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class TrackLossReason(str, Enum):
    """Auditable reason why a track is no longer directly observed."""

    UNMATCHED = "unmatched"
    EXPIRED = "expired"


class EventStatus(str, Enum):
    """Lifecycle states for a risk event."""

    OPEN = "open"
    UPDATED = "updated"
    ACKNOWLEDGED = "acknowledged"
    CLOSED = "closed"


def _validate_aware_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A decoded video frame and its timing metadata."""

    source_id: str
    frame_id: int
    captured_at: datetime
    width: int
    height: int
    payload: Any = field(default=None, repr=False, compare=False)
    dropped_before: int = 0
    decoded_at: datetime | None = None
    source_pts_seconds: float | None = None
    decode_duration_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.dropped_before < 0:
            raise ValueError("dropped_before must be non-negative")
        _validate_aware_timestamp(self.captured_at, "captured_at")
        if self.decoded_at is not None:
            _validate_aware_timestamp(self.decoded_at, "decoded_at")
        if self.source_pts_seconds is not None and (
            not isfinite(self.source_pts_seconds) or self.source_pts_seconds < 0
        ):
            raise ValueError("source_pts_seconds must be finite and non-negative")
        if self.decode_duration_ms is not None and (
            not isfinite(self.decode_duration_ms) or self.decode_duration_ms < 0
        ):
            raise ValueError("decode_duration_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TelemetryPacket:
    """Time-aligned platform and gimbal telemetry."""

    source_id: str
    captured_at: datetime
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    platform_roll_deg: float
    platform_pitch_deg: float
    platform_yaw_deg: float
    gimbal_roll_deg: float = 0.0
    gimbal_pitch_deg: float = 0.0
    gimbal_yaw_deg: float = 0.0
    quality: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        _validate_aware_timestamp(self.captured_at, "captured_at")
        numeric_values = (
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_m,
            self.platform_roll_deg,
            self.platform_pitch_deg,
            self.platform_yaw_deg,
            self.gimbal_roll_deg,
            self.gimbal_pitch_deg,
            self.gimbal_yaw_deg,
            self.quality,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("telemetry values must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be within [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg must be within [-180, 180]")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class Detection:
    """A detector output independent of the underlying ML framework."""

    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    detector_id: str
    roi_id: str | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must satisfy x2 >= x1 and y2 >= y1")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not self.class_name.strip():
            raise ValueError("class_name must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not self.detector_id.strip():
            raise ValueError("detector_id must not be empty")


@dataclass(frozen=True, slots=True)
class TrackState:
    """A single state estimate from a multi-object track."""

    track_id: int
    frame_id: int
    captured_at: datetime
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float
    observation: ObservationKind
    velocity_xy_px_s: tuple[float, float] = (0.0, 0.0)
    covariance_xy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    lifecycle: TrackLifecycle = TrackLifecycle.CONFIRMED
    tracker_id: str | None = None
    age_frames: int = 1
    time_since_update: int = 0
    association_score: float | None = None
    lost_reason: TrackLossReason | None = None
    global_motion_affine: tuple[float, float, float, float, float, float] | None = None
    global_motion_quality: float | None = None
    global_motion_applied: bool = False
    global_motion_fallback_reason: str | None = None
    global_motion_source: str | None = None
    global_motion_visual_quality: float | None = None
    global_motion_prior_quality: float | None = None
    global_motion_fusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.frame_id < 0:
            raise ValueError("track_id and frame_id must be non-negative")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(isfinite(value) for value in self.bbox_xyxy) or x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must be finite and satisfy x2 >= x1 and y2 >= y1")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not all(isfinite(value) for value in self.velocity_xy_px_s + self.covariance_xy):
            raise ValueError("velocity and covariance values must be finite")
        if self.tracker_id is not None and not self.tracker_id.strip():
            raise ValueError("tracker_id must not be empty when provided")
        if self.age_frames <= 0:
            raise ValueError("age_frames must be positive")
        if self.time_since_update < 0:
            raise ValueError("time_since_update must be non-negative")
        if self.association_score is not None and not 0.0 <= self.association_score <= 1.0:
            raise ValueError("association_score must be within [0, 1]")
        if self.observation is ObservationKind.OBSERVED and self.lost_reason is not None:
            raise ValueError("observed track states must not have a lost_reason")
        if self.observation is ObservationKind.INFERRED and self.lost_reason is None:
            raise ValueError("inferred track states must have a lost_reason")
        if self.global_motion_affine is None:
            if (
                self.global_motion_quality is not None
                or self.global_motion_applied
                or self.global_motion_source is not None
                or self.global_motion_visual_quality is not None
                or self.global_motion_prior_quality is not None
                or self.global_motion_fusion_reason is not None
            ):
                raise ValueError("global-motion quality/applied state requires affine coefficients")
        else:
            if len(self.global_motion_affine) != 6 or not all(isfinite(value) for value in self.global_motion_affine):
                raise ValueError("global_motion_affine must contain six finite coefficients")
            if self.global_motion_quality is None or not 0.0 <= self.global_motion_quality <= 1.0:
                raise ValueError("global_motion_quality must be within [0, 1] when an affine is provided")
        if self.global_motion_applied and self.global_motion_fallback_reason is not None:
            raise ValueError("applied global motion cannot have a fallback reason")
        if self.global_motion_fallback_reason is not None and not self.global_motion_fallback_reason.strip():
            raise ValueError("global_motion_fallback_reason must not be empty")
        if self.global_motion_source is not None and not self.global_motion_source.strip():
            raise ValueError("global_motion_source must not be empty")
        for name, quality in (
            ("global_motion_visual_quality", self.global_motion_visual_quality),
            ("global_motion_prior_quality", self.global_motion_prior_quality),
        ):
            if quality is not None and not 0.0 <= quality <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.global_motion_fusion_reason is not None and not self.global_motion_fusion_reason.strip():
            raise ValueError("global_motion_fusion_reason must not be empty")
        _validate_aware_timestamp(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class GeoEstimate:
    """A geospatial estimate with explicit quality."""

    track_id: int
    latitude_deg: float | None
    longitude_deg: float | None
    range_m: float | None
    bearing_deg: float | None
    range_rate_m_s: float | None
    quality: float
    absolute: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be within [0, 1]")
        if self.absolute and (self.latitude_deg is None or self.longitude_deg is None or self.range_m is None):
            raise ValueError("absolute estimates require coordinates and range")
        if self.range_m is not None and self.range_m < 0:
            raise ValueError("range_m must be non-negative")


@dataclass(frozen=True, slots=True)
class RiskEvent:
    """An auditable track-level risk event."""

    event_id: str
    rule_id: str
    zone_id: str
    track_id: int
    opened_at: datetime
    updated_at: datetime
    status: EventStatus
    risk_score: float
    evidence_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.rule_id or not self.zone_id:
            raise ValueError("event_id, rule_id, and zone_id must not be empty")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be within [0, 1]")
        _validate_aware_timestamp(self.opened_at, "opened_at")
        _validate_aware_timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at must not precede opened_at")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Metadata required to reproduce and safely deploy a model artifact."""

    model_id: str
    version: str
    framework: str
    input_size: tuple[int, int]
    class_names: tuple[str, ...]
    artifact_sha256: str
    git_commit: str
    data_version: str

    def __post_init__(self) -> None:
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a 64-character SHA-256 digest")
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError("input_size dimensions must be positive")
        if not self.class_names:
            raise ValueError("class_names must not be empty")
