from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from seahunter.schemas import Detection, FramePacket
from seahunter.services import FrameResult, ParquetResultSink


def frame_result(frame_id: int, with_detection: bool) -> FrameResult:
    detections: tuple[Detection, ...] = ()
    if with_detection:
        detections = (
            Detection(
                bbox_xyxy=(1.0, 2.0, 10.0, 12.0),
                class_id=0,
                class_name="swimmer",
                confidence=0.8,
                detector_id="detector:v1",
            ),
        )
    return FrameResult(
        frame=FramePacket(
            source_id="flight-01",
            frame_id=frame_id,
            captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            width=64,
            height=48,
            source_pts_seconds=frame_id / 10,
        ),
        detections=detections,
        detector_id="detector:v1",
        dropped_before=0,
        inference_duration_ms=1.0,
    )


class ParquetSinkTests(unittest.TestCase):
    def test_parquet_sink_streams_detection_and_empty_frames(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "detections.parquet"
            sink = ParquetResultSink(output, batch_size=1)
            sink.write(frame_result(0, True))
            sink.write(frame_result(1, False))
            sink.close()

            table = pq.read_table(output)
            rows = table.to_pylist()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["class_name"], "swimmer")
            self.assertEqual(rows[0]["frame_detection_count"], 1)
            self.assertIsNone(rows[1]["class_id"])
            self.assertEqual(rows[1]["frame_detection_count"], 0)
            self.assertEqual(sink.frames_written, 2)
            self.assertEqual(sink.rows_written, 2)
            with self.assertRaises(RuntimeError):
                sink.write(frame_result(2, False))


if __name__ == "__main__":
    unittest.main()
