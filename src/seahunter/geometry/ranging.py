"""Quality-gated line-of-sight intersection with a local sea surface."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, radians, sqrt

from seahunter.schemas import AltitudeDatum, FramePacket, GeoEstimate, TelemetryPacket, TrackState

from .calibration import CameraCalibration, calibration_quality, image_pixel_to_camera_ray
from .coordinates import camera_offset_to_enu, camera_ray_to_enu, enu_offset_to_geodetic


@dataclass(frozen=True, slots=True)
class SeaSurfaceContext:
    """Sea height and uncertainty expressed in the telemetry altitude datum."""

    altitude_m: float
    altitude_datum: AltitudeDatum
    platform_altitude_sigma_m: float = 2.0
    tide_sigma_m: float = 0.5
    wave_sigma_m: float = 1.0
    attitude_sigma_deg: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.altitude_m,
            self.platform_altitude_sigma_m,
            self.tide_sigma_m,
            self.wave_sigma_m,
            self.attitude_sigma_deg,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("sea-surface values must be finite")
        if self.altitude_datum is AltitudeDatum.UNKNOWN:
            raise ValueError("sea-surface altitude datum cannot be unknown")
        if any(value < 0.0 for value in values[1:]):
            raise ValueError("sea-surface uncertainty values must be non-negative")


@dataclass(frozen=True, slots=True)
class SeaPlaneGeolocatorConfig:
    minimum_downward_component: float = 0.03
    maximum_range_m: float = 5_000.0
    minimum_absolute_quality: float = 0.25
    maximum_calibration_rms_px: float = 1.0
    range_sigma_scale_m: float = 50.0

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_downward_component < 1.0:
            raise ValueError("minimum_downward_component must be within (0, 1)")
        if self.maximum_range_m <= 0.0 or self.maximum_calibration_rms_px <= 0.0:
            raise ValueError("range and calibration RMS gates must be positive")
        if not 0.0 <= self.minimum_absolute_quality <= 1.0:
            raise ValueError("minimum_absolute_quality must be within [0, 1]")
        if self.range_sigma_scale_m <= 0.0:
            raise ValueError("range_sigma_scale_m must be positive")


class SeaPlaneGeolocator:
    """Project a track contact pixel onto a locally horizontal sea surface."""

    def __init__(
        self,
        calibration: CameraCalibration,
        config: SeaPlaneGeolocatorConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = config or SeaPlaneGeolocatorConfig()

    def estimate(
        self,
        frame: FramePacket,
        track: TrackState,
        telemetry: TelemetryPacket | None,
        sea_surface: SeaSurfaceContext | None,
        *,
        alignment_quality: float = 1.0,
    ) -> GeoEstimate:
        if track.frame_id != frame.frame_id:
            raise ValueError("track and frame IDs must match")
        if not 0.0 <= alignment_quality <= 1.0:
            raise ValueError("alignment_quality must be within [0, 1]")
        if telemetry is None:
            return _degraded(track, frame, "missing_telemetry")
        if sea_surface is None:
            return _degraded(track, frame, "missing_sea_surface")
        if telemetry.source_id != frame.source_id:
            return _degraded(track, frame, "telemetry_source_mismatch")
        if telemetry.altitude_datum is AltitudeDatum.UNKNOWN:
            return _degraded(track, frame, "unknown_platform_altitude_datum")
        if telemetry.altitude_datum is not sea_surface.altitude_datum:
            return _degraded(track, frame, "altitude_datum_mismatch")
        intrinsics = self.calibration.intrinsics
        if (frame.width, frame.height) != (intrinsics.width, intrinsics.height):
            return _degraded(track, frame, "calibration_resolution_mismatch")

        x1, _, x2, y2 = track.bbox_xyxy
        ray_camera = image_pixel_to_camera_ray(((x1 + x2) / 2.0, y2), intrinsics)
        ray_enu = camera_ray_to_enu(ray_camera, telemetry, self.calibration)
        if ray_enu[2] >= -self.config.minimum_downward_component:
            return _degraded(track, frame, "ray_above_or_near_horizon")
        camera_offset = camera_offset_to_enu(telemetry, self.calibration)
        height_above_sea = telemetry.altitude_m + camera_offset[2] - sea_surface.altitude_m
        if height_above_sea <= 0.0:
            return _degraded(track, frame, "camera_not_above_sea_surface")
        slant_range = height_above_sea / -ray_enu[2]
        if slant_range > self.config.maximum_range_m:
            return _degraded(track, frame, "range_gate_exceeded")

        target_offset = (
            camera_offset[0] + ray_enu[0] * slant_range,
            camera_offset[1] + ray_enu[1] * slant_range,
            0.0,
        )
        latitude, longitude, _ = enu_offset_to_geodetic(
            telemetry.latitude_deg,
            telemetry.longitude_deg,
            target_offset,
        )
        bearing = (degrees(atan2(target_offset[0], target_offset[1])) + 360.0) % 360.0
        height_sigma = sqrt(
            sea_surface.platform_altitude_sigma_m**2 + sea_surface.tide_sigma_m**2 + sea_surface.wave_sigma_m**2
        )
        angular_sigma = radians(sea_surface.attitude_sigma_deg) + (
            intrinsics.calibration_rms_px / min(intrinsics.fx_px, intrinsics.fy_px)
        )
        vertical_component = -ray_enu[2]
        range_sigma = sqrt(
            (height_sigma / vertical_component) ** 2 + (height_above_sea * angular_sigma / (vertical_component**2)) ** 2
        )
        horizontal_sigma = max(
            0.1,
            sqrt(
                (range_sigma * hypot(ray_enu[0], ray_enu[1])) ** 2
                + (height_above_sea * angular_sigma / vertical_component) ** 2
            ),
        )
        uncertainty_quality = 1.0 / (1.0 + range_sigma / self.config.range_sigma_scale_m)
        quality = (
            calibration_quality(self.calibration, maximum_rms_px=self.config.maximum_calibration_rms_px)
            * telemetry.quality
            * alignment_quality
            * uncertainty_quality
        )
        if quality < self.config.minimum_absolute_quality:
            return GeoEstimate(
                track_id=track.track_id,
                latitude_deg=None,
                longitude_deg=None,
                range_m=slant_range,
                bearing_deg=bearing,
                range_rate_m_s=None,
                quality=quality,
                absolute=False,
                captured_at=frame.captured_at,
                covariance_en_m2=(horizontal_sigma**2, 0.0, 0.0, horizontal_sigma**2),
                degraded_reason="absolute_quality_below_gate",
            )
        return GeoEstimate(
            track_id=track.track_id,
            latitude_deg=latitude,
            longitude_deg=longitude,
            range_m=slant_range,
            bearing_deg=bearing,
            range_rate_m_s=None,
            quality=quality,
            absolute=True,
            captured_at=frame.captured_at,
            covariance_en_m2=(horizontal_sigma**2, 0.0, 0.0, horizontal_sigma**2),
        )


def _degraded(track: TrackState, frame: FramePacket, reason: str) -> GeoEstimate:
    return GeoEstimate(
        track_id=track.track_id,
        latitude_deg=None,
        longitude_deg=None,
        range_m=None,
        bearing_deg=None,
        range_rate_m_s=None,
        quality=0.0,
        absolute=False,
        captured_at=frame.captured_at,
        degraded_reason=reason,
    )
