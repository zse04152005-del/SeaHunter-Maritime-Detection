from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from seahunter.geometry import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    ENUConstantVelocityEKF,
    ENUKinematicState,
    ENUPositionMeasurement,
    SeaPlaneGeolocator,
    SeaSurfaceContext,
    estimate_relative_motion,
)
from seahunter.schemas import (
    AltitudeDatum,
    FramePacket,
    MotionTrend,
    ObservationKind,
    TelemetryPacket,
    TrackState,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)
OPTICAL_TO_FRD = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
COVARIANCE_4X4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def calibration() -> CameraCalibration:
    return CameraCalibration(
        calibration_id="geo-camera-v1",
        camera_serial="serial",
        lens_id="lens",
        calibrated_at=START,
        intrinsics=CameraIntrinsics(
            width=200,
            height=100,
            fx_px=100.0,
            fy_px=100.0,
            cx_px=100.0,
            cy_px=50.0,
            calibration_rms_px=0.2,
        ),
        extrinsics=CameraExtrinsics(OPTICAL_TO_FRD),
        quality=0.9,
    )


def frame() -> FramePacket:
    return FramePacket(source_id="flight", frame_id=0, captured_at=START, width=200, height=100)


def track() -> TrackState:
    return TrackState(
        track_id=7,
        frame_id=0,
        captured_at=START,
        bbox_xyxy=(90.0, 30.0, 110.0, 50.0),
        class_id=0,
        confidence=0.9,
        observation=ObservationKind.OBSERVED,
    )


def telemetry(*, gimbal_pitch_deg: float = -45.0, datum: AltitudeDatum = AltitudeDatum.AMSL) -> TelemetryPacket:
    return TelemetryPacket(
        source_id="flight",
        captured_at=START,
        latitude_deg=30.0,
        longitude_deg=120.0,
        altitude_m=100.0,
        platform_roll_deg=0.0,
        platform_pitch_deg=0.0,
        platform_yaw_deg=0.0,
        gimbal_pitch_deg=gimbal_pitch_deg,
        altitude_datum=datum,
    )


class GeolocationTests(unittest.TestCase):
    def test_downward_ray_intersects_sea_with_absolute_quality(self) -> None:
        estimate = SeaPlaneGeolocator(calibration()).estimate(
            frame(),
            track(),
            telemetry(),
            SeaSurfaceContext(altitude_m=0.0, altitude_datum=AltitudeDatum.AMSL),
        )

        self.assertTrue(estimate.absolute)
        self.assertAlmostEqual(estimate.range_m or 0.0, 141.421, places=2)
        self.assertAlmostEqual(estimate.bearing_deg or 0.0, 0.0, places=3)
        self.assertIsNotNone(estimate.latitude_deg)
        self.assertGreater(estimate.quality, 0.25)
        self.assertIsNotNone(estimate.covariance_en_m2)

    def test_datum_mismatch_and_horizon_explicitly_degrade(self) -> None:
        geolocator = SeaPlaneGeolocator(calibration())
        mismatch = geolocator.estimate(
            frame(),
            track(),
            telemetry(datum=AltitudeDatum.RELATIVE_HOME),
            SeaSurfaceContext(altitude_m=0.0, altitude_datum=AltitudeDatum.AMSL),
        )
        horizon = geolocator.estimate(
            frame(),
            track(),
            telemetry(gimbal_pitch_deg=0.0),
            SeaSurfaceContext(altitude_m=0.0, altitude_datum=AltitudeDatum.AMSL),
        )
        self.assertFalse(mismatch.absolute)
        self.assertEqual(mismatch.degraded_reason, "altitude_datum_mismatch")
        self.assertEqual(horizon.degraded_reason, "ray_above_or_near_horizon")

    def test_wave_uncertainty_reduces_quality(self) -> None:
        geolocator = SeaPlaneGeolocator(calibration())
        calm = geolocator.estimate(
            frame(),
            track(),
            telemetry(),
            SeaSurfaceContext(0.0, AltitudeDatum.AMSL, wave_sigma_m=0.1),
        )
        rough = geolocator.estimate(
            frame(),
            track(),
            telemetry(),
            SeaSurfaceContext(0.0, AltitudeDatum.AMSL, wave_sigma_m=20.0),
        )
        self.assertLess(rough.quality, calm.quality)

    def test_ekf_smooths_position_velocity_and_resets_after_long_gap(self) -> None:
        ekf = ENUConstantVelocityEKF()
        first = ekf.update(ENUPositionMeasurement(1, START, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0), 0.9))
        second = ekf.update(
            ENUPositionMeasurement(1, START + timedelta(seconds=1), 10.0, 0.0, (1.0, 0.0, 0.0, 1.0), 0.9)
        )
        reset = ekf.update(
            ENUPositionMeasurement(1, START + timedelta(seconds=10), 20.0, 0.0, (1.0, 0.0, 0.0, 1.0), 0.9)
        )
        self.assertEqual(first.east_velocity_m_s, 0.0)
        self.assertGreater(second.east_velocity_m_s, 0.0)
        self.assertEqual(reset.east_velocity_m_s, 0.0)
        self.assertEqual(len(second.covariance), 16)

    def test_relative_trends_and_ttc_degrade_by_quality(self) -> None:
        approaching = ENUKinematicState(1, START, 100.0, 0.0, -10.0, 0.0, COVARIANCE_4X4, 0.9)
        accepted = estimate_relative_motion(approaching, sensor_position_en_m=(0.0, 0.0))
        self.assertEqual(accepted.trend, MotionTrend.APPROACHING)
        self.assertAlmostEqual(accepted.ttc_seconds or 0.0, 10.0)

        low_quality = ENUKinematicState(1, START, 100.0, 0.0, -10.0, 0.0, COVARIANCE_4X4, 0.2)
        degraded = estimate_relative_motion(low_quality, sensor_position_en_m=(0.0, 0.0))
        self.assertEqual(degraded.trend, MotionTrend.APPROACHING)
        self.assertIsNone(degraded.ttc_seconds)
        self.assertEqual(degraded.degraded_reason, "ttc_quality_below_gate")

        crossing = ENUKinematicState(1, START, 100.0, 0.0, 0.0, 5.0, COVARIANCE_4X4, 0.9)
        self.assertEqual(
            estimate_relative_motion(crossing, sensor_position_en_m=(0.0, 0.0)).trend,
            MotionTrend.CROSSING,
        )


if __name__ == "__main__":
    unittest.main()
