"""Deployability-constrained maritime perception innovations."""

from .ablation import AblationMetrics, AblationRun, AblationSummary, MetricSummary, summarize_ablations
from .confidence import JointConfidenceConfig, JointConfidenceEvidence, JointConfidenceResult, JointConfidenceUpdater
from .downsampling import (
    DownsampleOperatorManifest,
    DownsampleVariant,
    joint_spatial_frequency_downsample,
    operator_manifest,
)
from .routing import (
    DegradationObservation,
    RouteProfile,
    RoutingDecision,
    WeatherRouter,
    WeatherRouterConfig,
    needs_sim2real,
)
from .selective_roi import (
    PerceptionPlan,
    ResourceState,
    ROIProposal,
    ROISource,
    ScaleDetection,
    SelectiveROIConfig,
    SelectiveROIScheduler,
    fuse_scale_detections,
)
from .temporal import (
    TemporalFeatureFrame,
    TemporalMonitorConfig,
    TemporalMonitorOutput,
    TemporalSignalClass,
    TemporalSmallTargetMonitor,
)

__all__ = [
    "AblationMetrics",
    "AblationRun",
    "AblationSummary",
    "DegradationObservation",
    "DownsampleOperatorManifest",
    "DownsampleVariant",
    "JointConfidenceConfig",
    "JointConfidenceEvidence",
    "JointConfidenceResult",
    "JointConfidenceUpdater",
    "MetricSummary",
    "PerceptionPlan",
    "ROIProposal",
    "ROISource",
    "ResourceState",
    "RouteProfile",
    "RoutingDecision",
    "ScaleDetection",
    "SelectiveROIConfig",
    "SelectiveROIScheduler",
    "TemporalFeatureFrame",
    "TemporalMonitorConfig",
    "TemporalMonitorOutput",
    "TemporalSignalClass",
    "TemporalSmallTargetMonitor",
    "WeatherRouter",
    "WeatherRouterConfig",
    "fuse_scale_detections",
    "joint_spatial_frequency_downsample",
    "needs_sim2real",
    "operator_manifest",
    "summarize_ablations",
]
