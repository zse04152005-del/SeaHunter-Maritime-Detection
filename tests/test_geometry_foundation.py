from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.geometry import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    camera_ray_to_enu,
    enu_offset_to_geodetic,
    enu_to_ned,
    image_pixel_to_camera_ray,
    load_camera_calibration,
    ned_to_enu,
)
from seahunter.schemas import TelemetryPacket
from seahunter.tools.calibration_audit import main as calibration_audit_main

OPTICAL_TO_FRD = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def calibration() -> CameraCalibration:
    return CameraCalibration(
        calibration_id="camera-test-v1",
        camera_serial="serial-1",
        lens_id="lens-1",
        calibrated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
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


class GeometryFoundationTests(unittest.TestCase):
    def test_center_pixel_maps_from_optical_forward_to_north_at_zero_attitude(self) -> None:
        current = calibration()
        ray = image_pixel_to_camera_ray((100.0, 50.0), current.intrinsics)
        telemetry = TelemetryPacket(
            source_id="flight",
            captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            latitude_deg=30.0,
            longitude_deg=120.0,
            altitude_m=100.0,
            platform_roll_deg=0.0,
            platform_pitch_deg=0.0,
            platform_yaw_deg=0.0,
        )

        self.assertEqual(ray, (0.0, 0.0, 1.0))
        east, north, up = camera_ray_to_enu(ray, telemetry, current)
        self.assertAlmostEqual(east, 0.0)
        self.assertAlmostEqual(north, 1.0)
        self.assertAlmostEqual(up, 0.0)

    def test_enu_ned_round_trip_and_geodetic_east_offset(self) -> None:
        vector = (12.0, 34.0, 5.0)
        self.assertEqual(ned_to_enu(enu_to_ned(vector)), vector)
        latitude, longitude, altitude = enu_offset_to_geodetic(30.0, 120.0, (100.0, 0.0, 0.0))
        self.assertAlmostEqual(latitude, 30.0, places=4)
        self.assertGreater(longitude, 120.0)
        self.assertAlmostEqual(altitude, 0.0, delta=0.1)

    def test_extrinsic_rotation_rejects_reflection(self) -> None:
        with self.assertRaises(ValueError):
            CameraExtrinsics((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    def test_strict_manifest_loader_and_audit(self) -> None:
        source = Path("configs/geometry/camera.example.json")
        loaded = load_camera_calibration(source)
        self.assertEqual(loaded.calibration_id, "camera-example-v1")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            self.assertEqual(calibration_audit_main([str(source), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["accepted"])
        self.assertEqual(report["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
