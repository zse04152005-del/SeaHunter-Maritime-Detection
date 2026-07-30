"""Multi-object tracking and tracklet recovery package."""

from .appearance import (
    AppearanceEncoder,
    AppearanceObservation,
    AppearanceQualityConfig,
    AppearanceQualityGate,
    HistogramAppearanceEncoder,
    aggregate_template,
    cosine_similarity,
)
from .base import MultiObjectTracker, TelemetryAwareTracker
from .botsort import BoTSORTConfig, BoTSORTTracker
from .bytetrack import ByteTrackConfig, ByteTracker
from .global_motion import (
    GlobalMotionEstimate,
    GlobalMotionEstimator,
    SparseOpticalFlowConfig,
    SparseOpticalFlowGMC,
)
from .kalman import KalmanXYWH
from .mot import MOTChallengeWriter, TrackJsonlWriter
from .motion_fusion import MotionFusionConfig, fuse_global_motion
from .telemetry_motion import TelemetryMotionConfig, TelemetryMotionEstimator, TelemetryMotionPrior

__all__ = [
    "AppearanceEncoder",
    "AppearanceObservation",
    "AppearanceQualityConfig",
    "AppearanceQualityGate",
    "BoTSORTConfig",
    "BoTSORTTracker",
    "ByteTrackConfig",
    "ByteTracker",
    "GlobalMotionEstimate",
    "GlobalMotionEstimator",
    "HistogramAppearanceEncoder",
    "KalmanXYWH",
    "MOTChallengeWriter",
    "MotionFusionConfig",
    "MultiObjectTracker",
    "SparseOpticalFlowConfig",
    "SparseOpticalFlowGMC",
    "TelemetryAwareTracker",
    "TelemetryMotionConfig",
    "TelemetryMotionEstimator",
    "TelemetryMotionPrior",
    "TrackJsonlWriter",
    "aggregate_template",
    "cosine_similarity",
    "fuse_global_motion",
]
