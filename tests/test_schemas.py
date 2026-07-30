from __future__ import annotations

import unittest
from datetime import datetime, timezone

from seahunter.schemas import (
    Detection,
    FramePacket,
    GeoEstimate,
    InferenceMethod,
    ObservationKind,
    TrackLossReason,
    TrackState,
)


class SchemaTests(unittest.TestCase):
    def test_frame_requires_aware_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            FramePacket(
                source_id="drone-01",
                frame_id=1,
                captured_at=datetime(2026, 7, 30),
                width=1920,
                height=1080,
            )

    def test_frame_timing_metadata_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            FramePacket(
                source_id="drone-01",
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                width=1920,
                height=1080,
                source_pts_seconds=-0.1,
            )
        with self.assertRaises(ValueError):
            FramePacket(
                source_id="drone-01",
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                width=1920,
                height=1080,
                decode_duration_ms=float("nan"),
            )

    def test_valid_detection_and_track(self) -> None:
        detection = Detection(
            bbox_xyxy=(1.0, 2.0, 12.0, 18.0),
            class_id=0,
            class_name="swimmer",
            confidence=0.8,
            detector_id="seahunter-v1",
        )
        track = TrackState(
            track_id=7,
            frame_id=12,
            captured_at=datetime.now(timezone.utc),
            bbox_xyxy=detection.bbox_xyxy,
            class_id=detection.class_id,
            confidence=detection.confidence,
            observation=ObservationKind.OBSERVED,
        )
        self.assertEqual(track.track_id, 7)

    def test_absolute_geo_estimate_requires_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            GeoEstimate(
                track_id=1,
                latitude_deg=None,
                longitude_deg=None,
                range_m=10.0,
                bearing_deg=20.0,
                range_rate_m_s=-1.0,
                quality=0.8,
                absolute=True,
            )

    def test_observed_track_cannot_have_lost_reason(self) -> None:
        with self.assertRaises(ValueError):
            TrackState(
                track_id=1,
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                class_id=0,
                confidence=0.8,
                observation=ObservationKind.OBSERVED,
                lost_reason=TrackLossReason.UNMATCHED,
            )

    def test_inferred_track_requires_lost_reason(self) -> None:
        with self.assertRaises(ValueError):
            TrackState(
                track_id=1,
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                class_id=0,
                confidence=0.8,
                observation=ObservationKind.INFERRED,
            )

    def test_inferred_track_requires_method(self) -> None:
        with self.assertRaises(ValueError):
            TrackState(
                track_id=1,
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                class_id=0,
                confidence=0.8,
                observation=ObservationKind.INFERRED,
                lost_reason=TrackLossReason.UNMATCHED,
            )

    def test_observed_track_rejects_inference_method(self) -> None:
        with self.assertRaises(ValueError):
            TrackState(
                track_id=1,
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                class_id=0,
                confidence=0.8,
                observation=ObservationKind.OBSERVED,
                inference_method=InferenceMethod.INTERPOLATED,
            )

    def test_applied_global_motion_requires_coefficients_and_quality(self) -> None:
        with self.assertRaises(ValueError):
            TrackState(
                track_id=1,
                frame_id=1,
                captured_at=datetime.now(timezone.utc),
                bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                class_id=0,
                confidence=0.8,
                observation=ObservationKind.OBSERVED,
                global_motion_applied=True,
            )


if __name__ == "__main__":
    unittest.main()
