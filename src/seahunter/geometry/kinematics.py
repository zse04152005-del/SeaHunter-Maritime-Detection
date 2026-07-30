"""Relative trend and TTC estimation from filtered ENU kinematics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isfinite

from seahunter.schemas import GeoEstimate, MotionTrend

from .filtering import ENUKinematicState


@dataclass(frozen=True, slots=True)
class TrendConfig:
    stationary_speed_m_s: float = 0.5
    radial_rate_threshold_m_s: float = 0.5
    minimum_ttc_quality: float = 0.6
    maximum_ttc_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.stationary_speed_m_s < 0.0 or self.radial_rate_threshold_m_s < 0.0:
            raise ValueError("trend speed thresholds must be non-negative")
        if not 0.0 <= self.minimum_ttc_quality <= 1.0:
            raise ValueError("minimum_ttc_quality must be within [0, 1]")
        if self.maximum_ttc_seconds <= 0.0:
            raise ValueError("maximum_ttc_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RelativeMotionEstimate:
    track_id: int
    trend: MotionTrend
    range_m: float
    range_rate_m_s: float
    tangential_speed_m_s: float
    ttc_seconds: float | None
    quality: float
    degraded_reason: str | None

    def __post_init__(self) -> None:
        numeric = (self.range_m, self.range_rate_m_s, self.tangential_speed_m_s, self.quality)
        if not all(isfinite(value) for value in numeric):
            raise ValueError("relative motion values must be finite")
        if self.range_m < 0.0 or self.tangential_speed_m_s < 0.0:
            raise ValueError("range and tangential speed must be non-negative")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("relative motion quality must be within [0, 1]")


def estimate_relative_motion(
    target: ENUKinematicState,
    *,
    sensor_position_en_m: tuple[float, float],
    sensor_velocity_en_m_s: tuple[float, float] = (0.0, 0.0),
    config: TrendConfig | None = None,
) -> RelativeMotionEstimate:
    settings = config or TrendConfig()
    relative_position = (
        target.east_m - sensor_position_en_m[0],
        target.north_m - sensor_position_en_m[1],
    )
    relative_velocity = (
        target.east_velocity_m_s - sensor_velocity_en_m_s[0],
        target.north_velocity_m_s - sensor_velocity_en_m_s[1],
    )
    distance = hypot(*relative_position)
    if distance <= 1e-9:
        radial_rate = 0.0
        tangential_speed = hypot(*relative_velocity)
    else:
        radial_rate = (
            relative_position[0] * relative_velocity[0] + relative_position[1] * relative_velocity[1]
        ) / distance
        tangential_x = relative_velocity[0] - radial_rate * relative_position[0] / distance
        tangential_y = relative_velocity[1] - radial_rate * relative_position[1] / distance
        tangential_speed = hypot(tangential_x, tangential_y)

    relative_speed = hypot(*relative_velocity)
    if relative_speed <= settings.stationary_speed_m_s:
        trend = MotionTrend.STATIONARY
    elif radial_rate <= -settings.radial_rate_threshold_m_s:
        trend = MotionTrend.APPROACHING
    elif radial_rate >= settings.radial_rate_threshold_m_s:
        trend = MotionTrend.RECEDING
    else:
        trend = MotionTrend.CROSSING

    ttc: float | None = None
    degraded_reason: str | None = None
    if target.quality < settings.minimum_ttc_quality:
        degraded_reason = "ttc_quality_below_gate"
    elif radial_rate >= -settings.radial_rate_threshold_m_s:
        degraded_reason = "not_closing"
    else:
        candidate = distance / -radial_rate
        if candidate <= settings.maximum_ttc_seconds:
            ttc = candidate
        else:
            degraded_reason = "ttc_horizon_exceeded"
    return RelativeMotionEstimate(
        track_id=target.track_id,
        trend=trend,
        range_m=distance,
        range_rate_m_s=radial_rate,
        tangential_speed_m_s=tangential_speed,
        ttc_seconds=ttc,
        quality=target.quality,
        degraded_reason=degraded_reason,
    )


def enrich_geo_estimate(estimate: GeoEstimate, motion: RelativeMotionEstimate) -> GeoEstimate:
    if estimate.track_id != motion.track_id:
        raise ValueError("geo and motion track IDs must match")
    reason = estimate.degraded_reason or motion.degraded_reason
    return replace(
        estimate,
        range_rate_m_s=motion.range_rate_m_s,
        trend=motion.trend,
        ttc_seconds=motion.ttc_seconds,
        quality=min(estimate.quality, motion.quality),
        degraded_reason=reason,
    )
