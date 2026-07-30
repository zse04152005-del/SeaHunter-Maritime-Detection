from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from seahunter.schemas import Detection, FramePacket
from seahunter.tracking import ByteTrackConfig, ByteTracker, KalmanXYWH

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def frame(index: int) -> FramePacket:
    return FramePacket(
        source_id="association-flight",
        frame_id=index,
        captured_at=START + timedelta(seconds=index / 10),
        width=640,
        height=360,
    )


def detection(
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        bbox_xyxy=bbox,
        class_id=0,
        class_name="swimmer",
        confidence=confidence,
        detector_id="detector:v1",
    )


class MotionAssociationTests(unittest.TestCase):
    def test_mahalanobis_gate_rejects_implausible_size_jump_that_has_iou(self) -> None:
        common = {
            "first_match_iou_threshold": 0.03,
            "new_track_threshold": 0.7,
        }
        gated = ByteTracker(ByteTrackConfig(**common, motion_gating_enabled=True))
        ungated = ByteTracker(ByteTrackConfig(**common, motion_gating_enabled=False))
        initial = detection((100.0, 100.0, 120.0, 120.0))
        implausible = detection((60.0, 60.0, 160.0, 160.0))

        gated.update(frame(0), [initial])
        ungated.update(frame(0), [initial])
        gated_state = gated.update(frame(1), [implausible])[0]
        ungated_state = ungated.update(frame(1), [implausible])[0]

        self.assertEqual(gated_state.track_id, 2)
        self.assertEqual(gated_state.association_stage, "new")
        self.assertEqual(ungated_state.track_id, 1)
        self.assertEqual(ungated_state.association_stage, "high")
        self.assertIsNone(ungated_state.motion_gate_distance)
        self.assertEqual(gated.association_summary()["motion_pairs_gated"], 1)

    def test_high_then_low_stages_are_audited_with_motion_distance(self) -> None:
        tracker = ByteTracker(ByteTrackConfig())

        first = tracker.update(frame(0), [detection((100.0, 100.0, 120.0, 120.0))])[0]
        second = tracker.update(
            frame(1),
            [detection((101.0, 100.0, 121.0, 120.0), confidence=0.3)],
        )[0]

        self.assertEqual(first.association_stage, "new")
        self.assertEqual(second.association_stage, "low")
        self.assertIsNotNone(second.motion_gate_distance)
        self.assertLess(second.motion_gate_distance or 0.0, tracker.config.motion_gate_threshold)
        self.assertEqual(tracker.association_summary()["stage_matches"], {"low": 1, "new": 1})

    def test_kalman_gate_distance_is_finite_and_orders_near_before_far(self) -> None:
        kalman = KalmanXYWH()
        mean, covariance = kalman.initiate((100.0, 100.0, 120.0, 120.0))
        predicted_mean, predicted_covariance = kalman.predict(mean, covariance)

        near = kalman.gating_distance(predicted_mean, predicted_covariance, (101.0, 100.0, 121.0, 120.0))
        far = kalman.gating_distance(predicted_mean, predicted_covariance, (300.0, 200.0, 340.0, 240.0))

        self.assertGreaterEqual(near, 0.0)
        self.assertLess(near, far)

    def test_reset_clears_association_diagnostics(self) -> None:
        tracker = ByteTracker(ByteTrackConfig())
        tracker.update(frame(0), [detection((100.0, 100.0, 120.0, 120.0))])
        self.assertEqual(tracker.association_summary()["stage_matches"], {"new": 1})

        tracker.reset()

        self.assertEqual(tracker.association_summary()["stage_matches"], {})
        self.assertEqual(tracker.association_summary()["motion_pairs_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
