"""Shared data contracts used across the SeaHunter-VIS pipeline."""

from .models import (
    AltitudeDatum,
    Detection,
    EventStatus,
    FramePacket,
    GeoEstimate,
    InferenceMethod,
    ModelManifest,
    MotionTrend,
    ObservationKind,
    RiskEvent,
    SourceKind,
    TelemetryPacket,
    TrackLifecycle,
    TrackLossReason,
    TrackState,
)

__all__ = [
    "AltitudeDatum",
    "Detection",
    "EventStatus",
    "FramePacket",
    "GeoEstimate",
    "InferenceMethod",
    "ModelManifest",
    "MotionTrend",
    "ObservationKind",
    "RiskEvent",
    "SourceKind",
    "TelemetryPacket",
    "TrackLifecycle",
    "TrackLossReason",
    "TrackState",
]
