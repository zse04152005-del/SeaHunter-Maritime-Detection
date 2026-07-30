from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.schemas import Detection, FramePacket
from seahunter.services import percentile, run_replay
from seahunter.video import ReaderStats, parse_video_source


class FakeReader:
    def __init__(self, frame_count: int) -> None:
        self.source = parse_video_source("flight.mp4")
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        self._frames = [
            FramePacket(
                source_id="flight-01",
                frame_id=index,
                captured_at=epoch + timedelta(seconds=index / 10),
                decoded_at=datetime(2026, 7, 30, tzinfo=UTC),
                source_pts_seconds=index / 10,
                decode_duration_ms=float(index + 1),
                width=64,
                height=48,
                payload=index,
            )
            for index in range(frame_count)
        ]
        self._decoded = 0
        self.closed = False

    def read(self) -> FramePacket | None:
        if not self._frames:
            return None
        self._decoded += 1
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True

    def stats(self) -> ReaderStats:
        return ReaderStats(
            source_id="flight-01",
            opened=not self.closed,
            open_count=1,
            open_failures=0,
            frames_decoded=self._decoded,
            read_failures=0,
            bad_frames=0,
            reconnect_attempts=0,
            last_error=None,
        )


class FakeDetector:
    @property
    def detector_id(self) -> str:
        return "fake-detector:v1"

    def infer(self, frame: FramePacket) -> list[Detection]:
        return [
            Detection(
                bbox_xyxy=(10.0, 8.0, 20.0, 18.0),
                class_id=1,
                class_name="boat",
                confidence=0.75,
                detector_id=self.detector_id,
            ),
            Detection(
                bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                class_id=0,
                class_name="swimmer",
                confidence=0.8,
                detector_id=self.detector_id,
            ),
        ]


class ReplayTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        with self.assertRaises(ValueError):
            percentile([1.0], 1.1)

    def test_offline_replay_metadata_is_byte_stable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first_output = root / "first.jsonl"
            second_output = root / "second.jsonl"
            first_summary = run_replay(FakeReader(3), FakeDetector(), first_output)
            second_summary = run_replay(FakeReader(3), FakeDetector(), second_output)

            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first_summary.metadata_sha256, second_summary.metadata_sha256)
            self.assertEqual(first_summary.frames_processed, 3)
            self.assertEqual(first_summary.detections_emitted, 6)

            records = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["captured_at"], "1970-01-01T00:00:00.000000Z")
            self.assertEqual(records[0]["detections"][0]["class_name"], "swimmer")
            self.assertNotIn("runtime_timing_ms", records[0])

    def test_replay_can_emit_summary_and_limit_frames(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "replay.jsonl"
            summary_path = root / "replay.summary.json"
            summary = run_replay(
                FakeReader(5),
                FakeDetector(),
                output,
                summary_path=summary_path,
                max_frames=2,
                include_runtime_timings=True,
            )

            self.assertEqual(summary.frames_processed, 2)
            self.assertTrue(summary_path.is_file())
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("runtime_timing_ms", first)


if __name__ == "__main__":
    unittest.main()
