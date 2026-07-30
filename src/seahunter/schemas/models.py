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
        _validate_aware_timestamp(self.captured_at, "captured_at")
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

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.frame_id < 0:
            raise ValueError("track_id and frame_id must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
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
