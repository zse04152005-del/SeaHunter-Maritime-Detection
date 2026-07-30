from __future__ import annotations

import importlib
import unittest
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from seahunter.schemas import Detection, FramePacket
from seahunter.tracking import (
    AppearanceQualityConfig,
    AppearanceQualityGate,
    BoTSORTConfig,
    BoTSORTTracker,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


class CountingEncoder:
    encoder_id = "counting-encoder:v1"

    def __init__(self, embedding: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.embedding = embedding
        self.calls = 0

    def encode(self, frame: FramePacket, detection: Detection, crop: Any) -> Sequence[float]:
        del frame, detection, crop
        self.calls += 1
        return self.embedding


def frame(index: int, *, value: int = 100) -> FramePacket:
    np: Any = importlib.import_module("numpy")
    return FramePacket(
        source_id="reid-flight",
        frame_id=index,
        captured_at=START + timedelta(seconds=index / 10),
        width=200,
        height=120,
        payload=np.full((120, 200, 3), value, dtype=np.uint8),
    )


def detection(x: float, *, size: float = 40.0) -> Detection:
    return Detection(
        bbox_xyxy=(x, 40.0, x + size, 40.0 + size),
        class_id=0,
        class_name="swimmer",
        confidence=0.9,
        detector_id="detector:v1",
    )


def permissive_quality() -> AppearanceQualityConfig:
    return AppearanceQualityConfig(
        minimum_short_side_px=10.0,
        minimum_area_px=100.0,
        minimum_visible_fraction=0.8,
        maximum_overlap_fraction=0.6,
        minimum_brightness=0.0,
        maximum_brightness=255.0,
        minimum_sharpness=0.0,
    )


class ReIDQualityTests(unittest.TestCase):
    def test_tiny_target_bypasses_encoder_before_crop_embedding(self) -> None:
        encoder = CountingEncoder()
        gate = AppearanceQualityGate(AppearanceQualityConfig(), encoder=encoder)

        observation = gate.observe(frame(0), detection(20.0, size=8.0), [detection(20.0, size=8.0)])

        self.assertFalse(observation.eligible)
        self.assertEqual(observation.bypass_reason, "target_too_small")
        self.assertEqual(encoder.calls, 0)

    def test_clear_tracklet_template_recovers_identity_after_iou_failure(self) -> None:
        encoder = CountingEncoder()
        tracker = BoTSORTTracker(
            BoTSORTConfig(
                gmc_enabled=False,
                reid_enabled=True,
                appearance_quality=permissive_quality(),
                minimum_appearance_similarity=0.9,
                reid_motion_gate_threshold=100_000.0,
                maximum_reid_age_frames=10,
                template_update_rate=0.5,
            ),
            appearance_encoder=encoder,
        )

        first = tracker.update(frame(0), [detection(20.0)])[0]
        second = tracker.update(frame(1), [detection(22.0)])[0]
        tracker.update(frame(2), [])
        recovered = tracker.update(frame(3), [detection(120.0)])[0]

        self.assertEqual((first.track_id, second.track_id, recovered.track_id), (1, 1, 1))
        self.assertEqual(recovered.association_stage, "reid")
        self.assertAlmostEqual(recovered.appearance_score or 0.0, 1.0)
        self.assertTrue(recovered.reid_eligible)
        self.assertIsNone(recovered.reid_bypass_reason)
        summary = tracker.appearance_summary()
        self.assertEqual(summary["template_updates"], 3)
        self.assertEqual(summary["reid_matches"], 1)

    def test_tiny_reappearance_cannot_hijack_lost_track(self) -> None:
        encoder = CountingEncoder()
        tracker = BoTSORTTracker(
            BoTSORTConfig(
                gmc_enabled=False,
                reid_enabled=True,
                reid_motion_gate_threshold=100_000.0,
            ),
            appearance_encoder=encoder,
        )

        first = tracker.update(frame(0), [detection(20.0, size=8.0)])[0]
        tracker.update(frame(1), [])
        second = tracker.update(frame(2), [detection(120.0, size=8.0)])[0]

        self.assertEqual(first.track_id, 1)
        self.assertEqual(second.track_id, 2)
        self.assertFalse(second.reid_eligible)
        self.assertEqual(second.reid_bypass_reason, "target_too_small")
        self.assertEqual(encoder.calls, 0)
        self.assertEqual(tracker.appearance_summary()["reid_matches"], 0)

    def test_overlap_gate_blocks_ambiguous_crop(self) -> None:
        encoder = CountingEncoder()
        gate = AppearanceQualityGate(permissive_quality(), encoder=encoder)
        target = detection(20.0)
        overlapping = detection(25.0)

        observation = gate.observe(frame(0), target, [target, overlapping])

        self.assertFalse(observation.eligible)
        self.assertEqual(observation.bypass_reason, "occluded_overlap")
        self.assertEqual(encoder.calls, 0)

    def test_low_confidence_match_does_not_contaminate_template(self) -> None:
        encoder = CountingEncoder()
        tracker = BoTSORTTracker(
            BoTSORTConfig(
                gmc_enabled=False,
                reid_enabled=True,
                appearance_quality=permissive_quality(),
            ),
            appearance_encoder=encoder,
        )
        low_confidence = Detection(
            bbox_xyxy=(22.0, 40.0, 62.0, 80.0),
            class_id=0,
            class_name="swimmer",
            confidence=0.4,
            detector_id="detector:v1",
        )

        tracker.update(frame(0), [detection(20.0)])
        state = tracker.update(frame(1), [low_confidence])[0]

        self.assertEqual(state.association_stage, "low")
        self.assertEqual(tracker.appearance_summary()["eligible_observations"], 2)
        self.assertEqual(tracker.appearance_summary()["template_updates"], 1)


if __name__ == "__main__":
    unittest.main()
