from __future__ import annotations

import unittest
from datetime import datetime, timezone

import cv2
import numpy as np

from seahunter.schemas import (
    Detection,
    FramePacket,
    ObservationKind,
    TrackLifecycle,
    TrackLossReason,
    TrackState,
)
from seahunter.services import FrameResult, JpegPreviewSink, PreviewHub, TrackingSink, TrackOverlayRenderer
from seahunter.tracking import ByteTracker


def result(
    frame_id: int,
    *,
    source_id: str = "camera-01",
    detections: tuple[Detection, ...] = (),
) -> FrameResult:
    return FrameResult(
        frame=FramePacket(
            source_id=source_id,
            frame_id=frame_id,
            captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            width=120,
            height=80,
            payload=np.zeros((80, 120, 3), dtype=np.uint8),
        ),
        detections=detections,
        detector_id="detector:v1",
        dropped_before=0,
        inference_duration_ms=2.0,
    )


def detection() -> Detection:
    return Detection(
        bbox_xyxy=(20.0, 20.0, 50.0, 50.0),
        class_id=1,
        class_name="boat",
        confidence=0.9,
        detector_id="detector:v1",
    )


def track(
    frame_id: int,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    observation: ObservationKind = ObservationKind.OBSERVED,
) -> TrackState:
    inferred = observation is ObservationKind.INFERRED
    return TrackState(
        track_id=7,
        frame_id=frame_id,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        bbox_xyxy=bbox_xyxy,
        class_id=1,
        confidence=0.75 if inferred else 0.9,
        observation=observation,
        lifecycle=TrackLifecycle.LOST if inferred else TrackLifecycle.CONFIRMED,
        tracker_id="bytetrack:test",
        age_frames=frame_id + 1,
        time_since_update=1 if inferred else 0,
        lost_reason=TrackLossReason.UNMATCHED if inferred else None,
    )


class TrackingVisualizationTests(unittest.TestCase):
    def test_renderer_draws_bounded_trail_without_mutating_source(self) -> None:
        renderer = TrackOverlayRenderer(trail_length=3)
        first_result = result(0)
        renderer.render(first_result, [track(0, (20.0, 20.0, 50.0, 50.0))])
        second_result = result(1)
        rendered = renderer.render(second_result, [track(1, (50.0, 20.0, 80.0, 50.0))])

        self.assertGreater(int(rendered[35, 45].sum()), 0)
        self.assertEqual(int(first_result.frame.payload.sum()), 0)
        self.assertEqual(int(second_result.frame.payload.sum()), 0)

    def test_inferred_state_uses_visible_dashed_prediction_style(self) -> None:
        renderer = TrackOverlayRenderer(trail_length=3)
        renderer.render(result(0), [track(0, (20.0, 20.0, 50.0, 50.0))])
        rendered = renderer.render(
            result(1),
            [track(1, (50.0, 20.0, 80.0, 50.0), observation=ObservationKind.INFERRED)],
        )

        self.assertGreater(int(rendered[20, 50, 2]), 0)

    def test_renderer_rejects_track_state_from_another_frame(self) -> None:
        renderer = TrackOverlayRenderer()
        with self.assertRaises(ValueError):
            renderer.render(result(1), [track(0, (20.0, 20.0, 50.0, 50.0))])

    def test_preview_publishes_track_metadata_and_rendered_jpeg(self) -> None:
        states = (track(0, (20.0, 20.0, 50.0, 50.0)),)
        hub = PreviewHub()
        sink = JpegPreviewSink(hub, track_state_provider=lambda: states)
        sink.write(result(0))
        latest = hub.snapshot()

        self.assertIsNotNone(latest)
        assert latest is not None
        decoded = cv2.imdecode(np.frombuffer(latest.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(int(decoded.sum()), 0)
        self.assertEqual(latest.metadata["track_count"], 1)
        tracks = latest.metadata["tracks"]
        self.assertIsInstance(tracks, list)
        assert isinstance(tracks, list)
        self.assertEqual(tracks[0]["track_id"], 7)
        self.assertEqual(tracks[0]["observation"], "observed")
        self.assertIsNone(tracks[0]["global_motion"])
        sink.close()

    def test_tracking_sink_snapshot_drives_same_frame_preview(self) -> None:
        tracking = TrackingSink(ByteTracker())
        hub = PreviewHub()
        preview = JpegPreviewSink(hub, track_state_provider=tracking.snapshot)
        frame_result = result(0, detections=(detection(),))

        tracking.write(frame_result)
        preview.write(frame_result)
        latest = hub.snapshot()

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.metadata["track_count"], 1)
        preview.close()
        tracking.close()

    def test_trail_length_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            TrackOverlayRenderer(trail_length=-1)


if __name__ == "__main__":
    unittest.main()
