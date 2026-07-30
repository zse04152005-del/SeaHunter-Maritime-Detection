from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.schemas import Detection, FramePacket
from seahunter.services import (
    AnnotatedVideoSink,
    JpegPreviewSink,
    ParquetResultSink,
    PreviewHub,
    run_replay,
)
from seahunter.video import OpenCVFrameReader, OpenCVReaderConfig, parse_video_source

HAS_VIDEO_STACK = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


class EmptyDetector:
    @property
    def detector_id(self) -> str:
        return "empty-detector:v1"

    def infer(self, frame: FramePacket) -> list[Detection]:
        del frame
        return []


@unittest.skipUnless(HAS_VIDEO_STACK, "OpenCV and NumPy are required for synthetic-video replay")
class OpenCVReplayTests(unittest.TestCase):
    def test_synthetic_video_end_to_end(self) -> None:
        import cv2
        import numpy as np

        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.avi"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("the installed OpenCV build cannot create MJPG video")
            for index in range(6):
                frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            reader = OpenCVFrameReader(
                parse_video_source(str(video)),
                source_id="synthetic",
                config=OpenCVReaderConfig(backend="ffmpeg", prefer_hardware_decode=False),
            )
            output = root / "synthetic.jsonl"
            annotated = root / "synthetic-annotated.avi"
            parquet = root / "synthetic.parquet"
            hub = PreviewHub()
            summary = run_replay(
                reader,
                EmptyDetector(),
                output,
                sinks=[
                    AnnotatedVideoSink(annotated, fps=10.0),
                    ParquetResultSink(parquet, batch_size=2),
                    JpegPreviewSink(hub),
                ],
            )

            self.assertEqual(summary.frames_processed, 6)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(record["width"], record["height"]) for record in records], [(64, 48)] * 6)
            pts = [record["source_pts_seconds"] for record in records]
            self.assertEqual(pts, sorted(pts))
            self.assertTrue(annotated.is_file())
            self.assertTrue(parquet.is_file())
            self.assertIsNotNone(hub.snapshot())
            self.assertTrue(hub.closed)


if __name__ == "__main__":
    unittest.main()
