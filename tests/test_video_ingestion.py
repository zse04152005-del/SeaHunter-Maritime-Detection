from __future__ import annotations

import unittest
from datetime import datetime, timezone

from seahunter.schemas import FramePacket
from seahunter.video import BufferClosed, FrameIngestWorker, IngestionFailed, parse_video_source


def make_frame(frame_id: int) -> FramePacket:
    return FramePacket(
        source_id="test-source",
        frame_id=frame_id,
        captured_at=datetime.now(timezone.utc),
        width=64,
        height=48,
        payload=object(),
    )


class FakeReader:
    def __init__(self, frames: list[FramePacket], *, failure: BaseException | None = None) -> None:
        self.source = parse_video_source("0")
        self._frames = list(frames)
        self._failure = failure
        self.closed = False

    def read(self) -> FramePacket | None:
        if self._frames:
            return self._frames.pop(0)
        if self._failure is not None:
            raise self._failure
        return None

    def close(self) -> None:
        self.closed = True


class VideoIngestionTests(unittest.TestCase):
    def test_fast_producer_retains_only_latest_frames(self) -> None:
        reader = FakeReader([make_frame(index) for index in range(5)])
        worker = FrameIngestWorker(reader, capacity=2)
        worker.start()
        self.assertTrue(worker.join(timeout=1.0))

        self.assertEqual(worker.get().frame_id, 3)
        self.assertEqual(worker.get().frame_id, 4)
        with self.assertRaises(BufferClosed):
            worker.get()
        stats = worker.stats()
        self.assertEqual(stats.produced, 5)
        self.assertEqual(stats.buffer.dropped, 3)
        self.assertTrue(reader.closed)

    def test_producer_failure_is_surfaced_to_consumer(self) -> None:
        worker = FrameIngestWorker(FakeReader([], failure=RuntimeError("decoder crashed")), capacity=1)
        worker.start()
        self.assertTrue(worker.join(timeout=1.0))
        with self.assertRaises(IngestionFailed):
            worker.get()

    def test_worker_cannot_be_started_twice(self) -> None:
        worker = FrameIngestWorker(FakeReader([]), capacity=1)
        worker.start()
        worker.join(timeout=1.0)
        with self.assertRaises(RuntimeError):
            worker.start()


if __name__ == "__main__":
    unittest.main()
