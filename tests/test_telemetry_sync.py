from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from seahunter.schemas import FramePacket, TelemetryPacket
from seahunter.telemetry import (
    ClockSyncConfig,
    ClockSynchronizer,
    FrameTelemetryAligner,
    FrameTelemetryAlignmentConfig,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def packet(seconds: float, *, quality: float = 1.0) -> TelemetryPacket:
    return TelemetryPacket(
        source_id="flight",
        captured_at=START + timedelta(seconds=seconds),
        latitude_deg=30.0,
        longitude_deg=120.0,
        altitude_m=100.0,
        platform_roll_deg=0.0,
        platform_pitch_deg=0.0,
        platform_yaw_deg=0.0,
        quality=quality,
    )


class TelemetrySyncTests(unittest.TestCase):
    def test_affine_clock_estimates_offset_drift_and_alignment(self) -> None:
        synchronizer = ClockSynchronizer(
            ClockSyncConfig(minimum_samples=3, maximum_absolute_drift_ppm=500.0, maximum_rmse_ms=5.0)
        )
        for source_seconds in (0.0, 1.0, 2.0, 3.0):
            synchronizer.add_sample(
                source_seconds,
                START + timedelta(seconds=10.0 + source_seconds * 1.0001),
            )

        estimate = synchronizer.estimate()
        self.assertTrue(estimate.accepted)
        self.assertAlmostEqual(estimate.drift_ppm or 0.0, 100.0, delta=0.1)
        aligned = synchronizer.align(4.0)
        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertAlmostEqual((aligned - START).total_seconds(), 14.0004, places=4)

    def test_backward_clock_jump_resets_window(self) -> None:
        synchronizer = ClockSynchronizer(ClockSyncConfig(minimum_samples=2))
        synchronizer.add_sample(10.0, START)
        synchronizer.add_sample(11.0, START + timedelta(seconds=1))
        self.assertTrue(synchronizer.estimate().accepted)
        estimate = synchronizer.add_sample(1.0, START + timedelta(seconds=2))
        self.assertFalse(estimate.accepted)
        self.assertEqual(estimate.reset_count, 1)

    def test_frame_alignment_applies_video_latency_and_gates_error(self) -> None:
        aligner = FrameTelemetryAligner(
            FrameTelemetryAlignmentConfig(video_latency_seconds=0.2, maximum_alignment_error_seconds=0.05)
        )
        aligner.add(packet(0.8))
        frame = FramePacket(
            source_id="flight",
            frame_id=1,
            captured_at=START + timedelta(seconds=1.0),
            width=1920,
            height=1080,
        )
        accepted = aligner.align(frame)
        self.assertTrue(accepted.accepted)
        self.assertAlmostEqual(accepted.alignment_error_seconds or 0.0, 0.0)

        stale = FrameTelemetryAligner(FrameTelemetryAlignmentConfig(maximum_alignment_error_seconds=0.05))
        stale.add(packet(0.0))
        self.assertEqual(stale.align(frame).reason, "alignment_error_exceeded")


if __name__ == "__main__":
    unittest.main()
