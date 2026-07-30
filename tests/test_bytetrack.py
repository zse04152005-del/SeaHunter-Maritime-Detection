from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from seahunter.schemas import (
    Detection,
    FramePacket,
    InferenceMethod,
    ObservationKind,
    TrackLifecycle,
    TrackLossReason,
)
from seahunter.tracking import ByteTrackConfig, ByteTracker


def frame(frame_id: int, *, source_id: str = "flight-01") -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame_id=frame_id,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc) + timedelta(seconds=frame_id / 10),
        width=640,
        height=360,
    )


def detection(
    x: float,
    *,
    confidence: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(
        bbox_xyxy=(x, 100.0, x + 20.0, 120.0),
        class_id=class_id,
        class_name="swimmer" if class_id == 0 else "boat",
        confidence=confidence,
        detector_id="detector:v1",
    )


class ByteTrackerTests(unittest.TestCase):
    def test_high_confidence_detection_keeps_stable_identity(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(frame_rate=10.0))
        first = tracker.update(frame(0), [detection(100.0)])
        second = tracker.update(frame(1), [detection(102.0)])

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)
        self.assertEqual(second[0].observation, ObservationKind.OBSERVED)
        self.assertEqual(second[0].lifecycle, TrackLifecycle.CONFIRMED)
        self.assertGreater(second[0].association_score or 0.0, 0.5)
        self.assertTrue(second[0].tracker_id and second[0].tracker_id.startswith("bytetrack:v1"))

    def test_low_confidence_detection_continues_but_does_not_create_track(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(frame_rate=10.0))
        self.assertEqual(tracker.update(frame(0), [detection(100.0, confidence=0.3)]), [])

        first = tracker.update(frame(1), [detection(100.0)])
        second = tracker.update(frame(2), [detection(101.0, confidence=0.3)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertAlmostEqual(second[0].confidence, 0.3)

    def test_short_occlusion_emits_inferred_state_and_recovers_same_id(self) -> None:
        tracker = ByteTracker(
            ByteTrackConfig(
                frame_rate=10.0,
                max_lost_frames=2,
                emit_lost_predictions=True,
            )
        )
        observed = tracker.update(frame(0), [detection(100.0)])[0]
        inferred = tracker.update(frame(1), [])[0]
        recovered = tracker.update(frame(2), [detection(101.0)])[0]

        self.assertEqual(observed.track_id, inferred.track_id)
        self.assertEqual(observed.track_id, recovered.track_id)
        self.assertEqual(inferred.observation, ObservationKind.INFERRED)
        self.assertEqual(inferred.lifecycle, TrackLifecycle.LOST)
        self.assertEqual(inferred.lost_reason, TrackLossReason.UNMATCHED)
        self.assertEqual(inferred.inference_method, InferenceMethod.EXTRAPOLATED)
        self.assertEqual(recovered.observation, ObservationKind.OBSERVED)
        self.assertIsNone(recovered.lost_reason)

    def test_expired_track_is_replaced_with_new_identity(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(max_lost_frames=1, emit_lost_predictions=True))
        self.assertEqual(tracker.update(frame(0), [detection(100.0)])[0].track_id, 1)
        self.assertEqual(tracker.update(frame(1), [])[0].track_id, 1)
        self.assertEqual(tracker.update(frame(2), []), [])
        self.assertEqual(tracker.update(frame(3), [detection(100.0)])[0].track_id, 2)

    def test_class_aware_matching_does_not_reuse_identity(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(emit_lost_predictions=True, class_aware=True))
        first = tracker.update(frame(0), [detection(100.0, class_id=0)])[0]
        states = tracker.update(frame(1), [detection(100.0, class_id=1)])
        observed = [state for state in states if state.observation is ObservationKind.OBSERVED]
        inferred = [state for state in states if state.observation is ObservationKind.INFERRED]

        self.assertEqual(first.track_id, 1)
        self.assertEqual(observed[0].track_id, 2)
        self.assertEqual(inferred[0].track_id, 1)

    def test_tentative_track_requires_configured_hits(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(minimum_confirmed_hits=2))
        self.assertEqual(tracker.update(frame(0), [detection(100.0)]), [])
        states = tracker.update(frame(1), [detection(101.0)])
        self.assertEqual(states[0].track_id, 1)
        self.assertEqual(states[0].age_frames, 2)

    def test_sequence_must_be_strictly_increasing_and_single_source(self) -> None:
        tracker = ByteTracker()
        tracker.update(frame(0), [detection(100.0)])
        with self.assertRaises(ValueError):
            tracker.update(frame(0), [detection(100.0)])
        with self.assertRaises(ValueError):
            tracker.update(frame(1, source_id="flight-02"), [detection(100.0)])

    def test_tracker_identifier_captures_behavioral_configuration(self) -> None:
        baseline = ByteTracker(ByteTrackConfig()).tracker_id
        changed = ByteTracker(
            ByteTrackConfig(
                frame_rate=25.0,
                inferred_confidence_decay=0.8,
                class_aware=False,
                emit_lost_predictions=True,
            )
        ).tracker_id
        self.assertNotEqual(baseline, changed)
        self.assertIn("fps=25", changed)
        self.assertIn("decay=0.8", changed)
        self.assertIn("class=0", changed)
        self.assertIn("emit_lost=1", changed)


if __name__ == "__main__":
    unittest.main()
