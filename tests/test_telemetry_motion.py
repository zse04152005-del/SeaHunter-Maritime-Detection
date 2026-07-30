from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from math import radians, tan

from seahunter.schemas import Detection, FramePacket, TelemetryPacket
from seahunter.services import FrameResult, TrackingSink
from seahunter.tracking import (
    BoTSORTConfig,
    BoTSORTTracker,
    GlobalMotionEstimate,
    MotionFusionConfig,
    TelemetryMotionConfig,
    TelemetryMotionPrior,
    fuse_global_motion,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def frame(index: int) -> FramePacket:
    return FramePacket(
        source_id="flight-telemetry",
        frame_id=index,
        captured_at=START + timedelta(seconds=index / 10),
        width=640,
        height=360,
    )


def telemetry(index: int, *, yaw: float, quality: float = 0.9) -> TelemetryPacket:
    return TelemetryPacket(
        source_id="flight-telemetry",
        captured_at=START + timedelta(seconds=index / 10),
        latitude_deg=30.0,
        longitude_deg=122.0,
        altitude_m=120.0,
        platform_roll_deg=0.0,
        platform_pitch_deg=0.0,
        platform_yaw_deg=yaw,
        quality=quality,
    )


def detection(x: float) -> Detection:
    return Detection(
        bbox_xyxy=(x, 100.0, x + 20.0, 120.0),
        class_id=0,
        class_name="swimmer",
        confidence=0.9,
        detector_id="detector:v1",
    )


def result(index: int, x: float, packet: TelemetryPacket) -> FrameResult:
    return FrameResult(
        frame=frame(index),
        detections=(detection(x),),
        detector_id="detector:v1",
        dropped_before=0,
        inference_duration_ms=1.0,
        telemetry=packet,
    )


def motion(x: float, *, quality: float, source: str) -> GlobalMotionEstimate:
    return GlobalMotionEstimate(
        affine_2x3=(1.0, 0.0, x, 0.0, 1.0, 0.0),
        quality=quality,
        feature_points=20 if source == "visual" else 0,
        tracked_points=18 if source == "visual" else 0,
        inliers=16 if source == "visual" else 0,
        applied=True,
        source=source,
    )


class TelemetryMotionTests(unittest.TestCase):
    def test_yaw_delta_becomes_opposite_scene_translation(self) -> None:
        prior = TelemetryMotionPrior(TelemetryMotionConfig(focal_length_px=1000.0))
        prior.add(telemetry(0, yaw=0.0))
        prior.add(telemetry(1, yaw=2.0))

        estimate = prior.estimate(frame(0), frame(1))

        self.assertTrue(estimate.applied)
        self.assertEqual(estimate.source, "telemetry")
        self.assertAlmostEqual(estimate.translation_xy[0], -1000.0 * tan(radians(2.0)), places=6)
        self.assertAlmostEqual(estimate.translation_xy[1], 0.0, places=6)
        self.assertEqual(estimate.prior_quality, 0.9)

    def test_yaw_wrap_uses_short_rotation_and_alignment_is_gated(self) -> None:
        prior = TelemetryMotionPrior(TelemetryMotionConfig(focal_length_px=1000.0))
        prior.add(telemetry(0, yaw=179.0))
        prior.add(telemetry(1, yaw=-179.0))

        estimate = prior.estimate(frame(0), frame(1))
        missing = prior.estimate(frame(10), frame(11))

        self.assertTrue(estimate.applied)
        self.assertAlmostEqual(estimate.translation_xy[0], -1000.0 * tan(radians(2.0)), places=6)
        self.assertFalse(missing.applied)
        self.assertEqual(missing.fallback_reason, "telemetry_alignment_miss")

    def test_motion_fusion_blends_compatible_affines_and_audits_disagreement(self) -> None:
        config = MotionFusionConfig(maximum_disagreement_ratio=0.03)
        compatible = fuse_global_motion(
            motion(-30.0, quality=0.8, source="visual"),
            motion(-32.0, quality=0.6, source="telemetry"),
            frame(1),
            config,
        )
        disagreement = fuse_global_motion(
            motion(40.0, quality=0.7, source="visual"),
            motion(-40.0, quality=0.9, source="telemetry"),
            frame(1),
            config,
        )

        self.assertEqual(compatible.source, "fused")
        self.assertEqual(compatible.fusion_reason, "compatible_weighted_affine")
        self.assertAlmostEqual(compatible.translation_xy[0], -30.8571428571)
        self.assertEqual(disagreement.source, "telemetry")
        self.assertEqual(disagreement.fusion_reason, "affine_disagreement_selected_telemetry")

    def test_tracking_sink_feeds_prior_and_preserves_identity_without_visual_gmc(self) -> None:
        prior = TelemetryMotionPrior(TelemetryMotionConfig(focal_length_px=1000.0))
        tracker = BoTSORTTracker(BoTSORTConfig(gmc_enabled=False), motion_prior=prior)
        sink = TrackingSink(tracker)
        shifted_x = 100.0 - 1000.0 * tan(radians(2.0))

        sink.write(result(0, 100.0, telemetry(0, yaw=0.0)))
        first = sink.snapshot()[0]
        sink.write(result(1, shifted_x, telemetry(1, yaw=2.0)))
        second = sink.snapshot()[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertTrue(second.global_motion_applied)
        self.assertEqual(second.global_motion_source, "telemetry")
        self.assertEqual(second.global_motion_fusion_reason, "visual_rejected:gmc_disabled")
        self.assertEqual(tracker.motion_summary()["source_counts"], {"telemetry": 1})


if __name__ == "__main__":
    unittest.main()
