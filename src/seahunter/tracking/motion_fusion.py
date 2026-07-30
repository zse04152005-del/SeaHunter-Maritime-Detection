"""Quality-gated fusion of visual GMC and telemetry camera-motion priors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import cast

from seahunter.schemas import FramePacket

from .global_motion import GlobalMotionEstimate


@dataclass(frozen=True, slots=True)
class MotionFusionConfig:
    """Controls for combining two independently gated affine estimates."""

    telemetry_weight: float = 1.0
    maximum_disagreement_ratio: float = 0.03

    def __post_init__(self) -> None:
        if self.telemetry_weight <= 0:
            raise ValueError("telemetry_weight must be positive")
        if not 0.0 <= self.maximum_disagreement_ratio <= 1.0:
            raise ValueError("maximum_disagreement_ratio must be within [0, 1]")


def fuse_global_motion(
    visual: GlobalMotionEstimate,
    prior: GlobalMotionEstimate,
    frame: FramePacket,
    config: MotionFusionConfig,
) -> GlobalMotionEstimate:
    """Fuse compatible estimates, otherwise choose the higher-quality source."""

    if visual.applied and not prior.applied:
        return replace(
            visual,
            visual_quality=visual.quality,
            prior_quality=prior.quality,
            fusion_reason=f"telemetry_rejected:{prior.fallback_reason}",
        )
    if prior.applied and not visual.applied:
        return replace(
            prior,
            visual_quality=visual.quality,
            prior_quality=prior.quality,
            fusion_reason=f"visual_rejected:{visual.fallback_reason}",
        )
    if not visual.applied and not prior.applied:
        return GlobalMotionEstimate.identity(
            f"visual={visual.fallback_reason};telemetry={prior.fallback_reason}",
            source="none",
            visual_quality=visual.quality,
            prior_quality=prior.quality,
            fusion_reason="both_sources_rejected",
        )

    disagreement = _maximum_corner_disagreement(visual.affine_2x3, prior.affine_2x3, frame)
    maximum_disagreement = config.maximum_disagreement_ratio * hypot(float(frame.width), float(frame.height))
    if disagreement <= maximum_disagreement:
        visual_weight = max(visual.quality, 1e-9)
        prior_weight = max(prior.quality * config.telemetry_weight, 1e-9)
        total_weight = visual_weight + prior_weight
        affine = cast(
            tuple[float, float, float, float, float, float],
            tuple(
                (visual_weight * visual_value + prior_weight * prior_value) / total_weight
                for visual_value, prior_value in zip(visual.affine_2x3, prior.affine_2x3, strict=True)
            ),
        )
        return GlobalMotionEstimate(
            affine_2x3=affine,
            quality=(visual_weight * visual.quality + prior_weight * prior.quality) / total_weight,
            feature_points=visual.feature_points,
            tracked_points=visual.tracked_points,
            inliers=visual.inliers,
            applied=True,
            source="fused",
            visual_quality=visual.quality,
            prior_quality=prior.quality,
            fusion_reason="compatible_weighted_affine",
        )

    if prior.quality * config.telemetry_weight > visual.quality:
        return replace(
            prior,
            visual_quality=visual.quality,
            prior_quality=prior.quality,
            fusion_reason="affine_disagreement_selected_telemetry",
        )
    return replace(
        visual,
        visual_quality=visual.quality,
        prior_quality=prior.quality,
        fusion_reason="affine_disagreement_selected_visual",
    )


def _maximum_corner_disagreement(first: tuple[float, ...], second: tuple[float, ...], frame: FramePacket) -> float:
    corners = (
        (0.0, 0.0),
        (float(frame.width), 0.0),
        (0.0, float(frame.height)),
        (float(frame.width), float(frame.height)),
    )
    maximum = 0.0
    for x, y in corners:
        first_x = first[0] * x + first[1] * y + first[2]
        first_y = first[3] * x + first[4] * y + first[5]
        second_x = second[0] * x + second[1] * y + second[2]
        second_y = second[3] * x + second[4] * y + second[5]
        maximum = max(maximum, hypot(first_x - second_x, first_y - second_y))
    return maximum
