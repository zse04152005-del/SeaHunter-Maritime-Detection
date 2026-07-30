from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from math import radians

from seahunter.schemas import AltitudeDatum
from seahunter.telemetry import ClockSyncConfig, ClockSynchronizer, MAVLinkTelemetryAdapter

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


class MAVLinkAdapterTests(unittest.TestCase):
    def test_position_and_attitude_are_aggregated_after_clock_gate(self) -> None:
        adapter = MAVLinkTelemetryAdapter(
            "flight",
            clock=ClockSynchronizer(
                ClockSyncConfig(minimum_samples=2, maximum_absolute_drift_ppm=100.0, maximum_rmse_ms=5.0)
            ),
        )
        position = {
            "mavpackettype": "GLOBAL_POSITION_INT",
            "time_boot_ms": 0,
            "lat": 300_000_000,
            "lon": 1_200_000_000,
            "relative_alt": 100_000,
            "alt": 110_000,
        }
        attitude = {
            "mavpackettype": "ATTITUDE",
            "time_boot_ms": 1000,
            "roll": radians(1.0),
            "pitch": radians(-2.0),
            "yaw": radians(90.0),
        }

        self.assertIsNone(adapter.ingest(position, START))
        result = adapter.ingest(attitude, START + timedelta(seconds=1))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.latitude_deg, 30.0)
        self.assertEqual(result.longitude_deg, 120.0)
        self.assertEqual(result.altitude_m, 100.0)
        self.assertEqual(result.altitude_datum, AltitudeDatum.RELATIVE_HOME)
        self.assertAlmostEqual(result.platform_pitch_deg, -2.0)
        self.assertAlmostEqual(result.platform_yaw_deg, 90.0)
        self.assertGreater(result.quality, 0.0)

    def test_missing_boot_clock_never_fabricates_timestamp(self) -> None:
        adapter = MAVLinkTelemetryAdapter("flight")
        self.assertIsNone(adapter.ingest({"mavpackettype": "ATTITUDE", "roll": 0, "pitch": 0, "yaw": 0}, START))


if __name__ == "__main__":
    unittest.main()
