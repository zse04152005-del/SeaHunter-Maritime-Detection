from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.schemas import Detection, FramePacket
from seahunter.tracking import (
    BoTSORTConfig,
    BoTSORTTracker,
    GlobalMotionEstimate,
    MOTChallengeWriter,
    TrackJsonlWriter,
)


def frame(frame_id: int) -> FramePacket:
    return FramePacket(
        source_id="flight-01",
        frame_id=frame_id,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc) + timedelta(seconds=frame_id / 10),
        width=640,
        height=360,
    )


def detection(x: float) -> Detection:
    return Detection(
        bbox_xyxy=(x, 100.0, x + 20.0, 120.0),
        class_id=0,
        class_name="swimmer",
        confidence=0.9,
        detector_id="detector:v1",
    )


class TranslationEstimator:
    estimator_id = "scripted-translation:v1"

    def __init__(self, x: float, y: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.calls = 0

    def estimate(
        self,
        previous_frame: FramePacket,
        current_frame: FramePacket,
        *,
        previous_detections: object = (),
        current_detections: object = (),
    ) -> GlobalMotionEstimate:
        del previous_frame, current_frame, previous_detections, current_detections
        self.calls += 1
        return GlobalMotionEstimate(
            affine_2x3=(1.0, 0.0, self.x, 0.0, 1.0, self.y),
            quality=0.9,
            feature_points=40,
            tracked_points=36,
            inliers=32,
            applied=True,
        )

    def reset(self) -> None:
        self.calls = 0


class BoTSORTTrackerTests(unittest.TestCase):
    def test_global_translation_preserves_identity_when_iou_alone_cannot(self) -> None:
        estimator = TranslationEstimator(40.0)
        tracker = BoTSORTTracker(BoTSORTConfig(), motion_estimator=estimator)

        first = tracker.update(frame(0), [detection(100.0)])[0]
        second = tracker.update(frame(1), [detection(140.0)])[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(estimator.calls, 1)
        self.assertTrue(second.global_motion_applied)
        self.assertEqual(second.global_motion_affine, (1.0, 0.0, 40.0, 0.0, 1.0, 0.0))
        self.assertEqual(second.global_motion_quality, 0.9)
        self.assertTrue(second.tracker_id and second.tracker_id.startswith("botsort:v1"))
        self.assertIn(":reid=0", second.tracker_id or "")

    def test_gmc_can_be_disabled_for_paired_ablation(self) -> None:
        tracker = BoTSORTTracker(BoTSORTConfig(gmc_enabled=False))
        first = tracker.update(frame(0), [detection(100.0)])[0]
        second = tracker.update(frame(1), [detection(140.0)])[0]

        self.assertNotEqual(first.track_id, second.track_id)
        self.assertFalse(second.global_motion_applied)
        self.assertEqual(second.global_motion_fallback_reason, "gmc_disabled")

    def test_motion_diagnostics_are_written_to_audit_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audit = TrackJsonlWriter(root / "tracks.jsonl")
            mot = MOTChallengeWriter(root / "tracks.txt")
            tracker = BoTSORTTracker(BoTSORTConfig(), motion_estimator=TranslationEstimator(40.0))

            first = tracker.update(frame(0), [detection(100.0)])
            second = tracker.update(frame(1), [detection(140.0)])
            audit.write(first)
            audit.write(second)
            mot.write(first)
            mot.write(second)
            audit.close()
            mot.close()

            rows = [json.loads(line) for line in (root / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["schema_version"], 4)
            self.assertEqual(rows[0]["global_motion"]["fallback_reason"], "no_previous_frame")
            self.assertTrue(rows[1]["global_motion"]["applied"])
            self.assertEqual(rows[1]["global_motion"]["affine_2x3"][2], 40.0)
            self.assertEqual(len((root / "tracks.txt").read_text(encoding="utf-8").splitlines()), 2)

    def test_reset_clears_identity_and_motion_counters(self) -> None:
        estimator = TranslationEstimator(40.0)
        tracker = BoTSORTTracker(BoTSORTConfig(), motion_estimator=estimator)
        tracker.update(frame(0), [detection(100.0)])
        tracker.update(frame(1), [detection(140.0)])
        self.assertEqual(tracker.motion_summary()["frames_applied"], 1)

        tracker.reset()
        self.assertEqual(tracker.motion_summary()["frames_applied"], 0)
        self.assertEqual(tracker.update(frame(0), [detection(100.0)])[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
