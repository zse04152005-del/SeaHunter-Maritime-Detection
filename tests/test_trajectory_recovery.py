from __future__ import annotations

import unittest
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from seahunter.schemas import Detection, FramePacket, InferenceMethod, TrackState
from seahunter.tracking import ByteTrackConfig, ByteTracker, TrajectoryRecoveryWriter

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def frame(index: int) -> FramePacket:
    return FramePacket(
        source_id="recovery-flight",
        frame_id=index,
        captured_at=START + timedelta(seconds=index / 10),
        width=640,
        height=360,
    )


def detection(x: float) -> Detection:
    return Detection(
        bbox_xyxy=(x, 40.0, x + 20.0, 60.0),
        class_id=0,
        class_name="swimmer",
        confidence=0.9,
        detector_id="detector:v1",
    )


class Collector:
    def __init__(self) -> None:
        self.states: list[TrackState] = []
        self.closed = False

    def write(self, states: Sequence[TrackState]) -> None:
        self.states.extend(states)

    def close(self) -> None:
        self.closed = True


class TrajectoryRecoveryTests(unittest.TestCase):
    def test_short_closed_gap_replaces_extrapolation_with_interpolation(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(emit_lost_predictions=True, motion_gating_enabled=False))
        collector = Collector()
        recovery = TrajectoryRecoveryWriter([collector], maximum_gap_frames=2)

        recovery.write(tracker.update(frame(0), [detection(10.0)]))
        recovery.write(tracker.update(frame(1), []))
        recovery.write(tracker.update(frame(2), [detection(14.0)]))
        recovery.close()

        self.assertTrue(collector.closed)
        self.assertEqual([state.frame_id for state in collector.states], [0, 1, 2])
        start, middle, end = collector.states
        expected = tuple((left + right) / 2.0 for left, right in zip(start.bbox_xyxy, end.bbox_xyxy, strict=True))
        self.assertEqual(middle.inference_method, InferenceMethod.INTERPOLATED)
        for actual, wanted in zip(middle.bbox_xyxy, expected, strict=True):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(recovery.summary()["interpolated_gaps"], 1)
        self.assertEqual(recovery.summary()["interpolated_points"], 1)
        self.assertEqual(recovery.summary()["extrapolated_points"], 0)

    def test_gap_beyond_bound_remains_finite_extrapolation(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(emit_lost_predictions=True, motion_gating_enabled=False))
        collector = Collector()
        recovery = TrajectoryRecoveryWriter([collector], maximum_gap_frames=1)

        recovery.write(tracker.update(frame(0), [detection(10.0)]))
        recovery.write(tracker.update(frame(1), []))
        recovery.write(tracker.update(frame(2), []))
        recovery.write(tracker.update(frame(3), [detection(14.0)]))
        recovery.close()

        inferred = [state for state in collector.states if state.inference_method is not None]
        self.assertEqual(len(inferred), 2)
        self.assertTrue(all(state.inference_method is InferenceMethod.EXTRAPOLATED for state in inferred))
        self.assertEqual(recovery.summary()["interpolated_gaps"], 0)
        self.assertEqual(recovery.summary()["extrapolated_points"], 2)

    def test_recovery_writer_rejects_non_monotonic_frames(self) -> None:
        tracker = ByteTracker(ByteTrackConfig(emit_lost_predictions=True))
        states = tracker.update(frame(0), [detection(10.0)])
        recovery = TrajectoryRecoveryWriter([Collector()], maximum_gap_frames=2)
        recovery.write(states)
        with self.assertRaises(ValueError):
            recovery.write(states)


if __name__ == "__main__":
    unittest.main()
