from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from seahunter.schemas import Detection, FramePacket
from seahunter.services import (
    AnnotatedVideoSink,
    FrameResult,
    JpegPreviewSink,
    PreviewHub,
    frame_result_to_record,
    render_annotated_frame,
)


def make_result(frame_id: int = 0) -> FrameResult:
    frame = FramePacket(
        source_id="camera-01",
        frame_id=frame_id,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        width=96,
        height=64,
        payload=np.zeros((64, 96, 3), dtype=np.uint8),
        source_pts_seconds=frame_id / 5,
        decode_duration_ms=2.0,
    )
    detection = Detection(
        bbox_xyxy=(10.0, 12.0, 50.0, 45.0),
        class_id=1,
        class_name="boat",
        confidence=0.9,
        detector_id="detector:v1",
    )
    return FrameResult(
        frame=frame,
        detections=(detection,),
        detector_id="detector:v1",
        dropped_before=0,
        inference_duration_ms=4.5,
    )


class ResultSinkTests(unittest.TestCase):
    def test_canonical_record_and_annotation_do_not_mutate_source(self) -> None:
        result = make_result()
        rendered = render_annotated_frame(result)
        record = frame_result_to_record(result, include_runtime_timings=True)

        self.assertGreater(int(rendered.sum()), 0)
        self.assertEqual(int(result.frame.payload.sum()), 0)
        self.assertEqual(record["frame_id"], 0)
        self.assertIn("runtime_timing_ms", record)

    def test_annotated_video_sink_writes_readable_video(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "annotated.avi"
            sink = AnnotatedVideoSink(output, fps=5.0)
            sink.write(make_result(0))
            sink.write(make_result(1))
            sink.close()

            capture = cv2.VideoCapture(str(output))
            frames = 0
            while capture.read()[0]:
                frames += 1
            capture.release()
            self.assertEqual(frames, 2)
            self.assertEqual(sink.frames_written, 2)
            with self.assertRaises(RuntimeError):
                sink.write(make_result(2))

    def test_jpeg_preview_sink_publishes_latest_frame(self) -> None:
        hub = PreviewHub()
        sink = JpegPreviewSink(hub, quality=75)
        sink.write(make_result())
        latest = hub.snapshot()

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertTrue(latest.jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(latest.metadata["frame_id"], 0)
        sink.close()
        self.assertTrue(hub.closed)


if __name__ == "__main__":
    unittest.main()
