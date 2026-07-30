from __future__ import annotations

import unittest
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from seahunter.video import (
    ExponentialBackoff,
    OpenCVFrameReader,
    OpenCVReaderConfig,
    StreamUnavailable,
    parse_video_source,
)


class FakeFrame:
    def __init__(self, height: int = 48, width: int = 64) -> None:
        self.shape = (height, width, 3)


class FakeCapture:
    def __init__(
        self,
        reads: list[tuple[bool, Any]],
        *,
        positions_ms: list[float] | None = None,
        fps: float = 25.0,
        opened: bool = True,
    ) -> None:
        self._reads = reads
        self._positions_ms = positions_ms or [0.0] * len(reads)
        self._fps = fps
        self._opened = opened
        self._read_index = 0
        self.released = False

    def is_opened(self) -> bool:
        return self._opened and not self.released

    def read(self) -> tuple[bool, Any]:
        result = self._reads[self._read_index]
        self._read_index += 1
        return result

    def position_ms(self) -> float:
        return self._positions_ms[max(0, self._read_index - 1)]

    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        self.released = True


class RaisingCapture(FakeCapture):
    def __init__(self) -> None:
        super().__init__([])

    def read(self) -> tuple[bool, Any]:
        raise RuntimeError("backend decoder failure")


def factory_from(captures: list[FakeCapture]) -> Callable[..., FakeCapture]:
    pending = list(captures)

    def factory(*args: object, **kwargs: object) -> FakeCapture:
        del args, kwargs
        return pending.pop(0)

    return factory


class VideoReaderTests(unittest.TestCase):
    def test_missing_local_file_is_checked_only_when_opening(self) -> None:
        reader = OpenCVFrameReader(parse_video_source("missing-flight.mp4"))
        with self.assertRaises(FileNotFoundError):
            reader.read()

    def test_file_pts_is_deterministic_and_monotonic(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "flight.avi"
            path.touch()
            capture = FakeCapture(
                [(True, FakeFrame()), (True, FakeFrame()), (False, None)],
                positions_ms=[0.0, 0.0, 0.0],
                fps=25.0,
            )
            reader = OpenCVFrameReader(
                parse_video_source(str(path)),
                capture_factory=factory_from([capture]),
                utc_clock=lambda: datetime(2026, 7, 30, tzinfo=UTC),
            )

            first = reader.read()
            second = reader.read()
            end = reader.read()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            assert first is not None and second is not None
            self.assertEqual(first.source_pts_seconds, 0.0)
            self.assertAlmostEqual(second.source_pts_seconds or 0.0, 0.04)
            self.assertEqual(first.captured_at, datetime(1970, 1, 1, tzinfo=UTC))
            self.assertEqual((first.width, first.height), (64, 48))
            self.assertIsNone(end)
            self.assertEqual(reader.stats().frames_decoded, 2)

    def test_live_read_failure_reconnects_and_preserves_frame_sequence(self) -> None:
        failed = FakeCapture([(False, None)])
        recovered = FakeCapture([(True, FakeFrame())], positions_ms=[123.0])
        config = OpenCVReaderConfig(
            max_reconnect_attempts=1,
            backoff=ExponentialBackoff(initial_seconds=0.0, maximum_seconds=0.0),
        )
        reader = OpenCVFrameReader(
            parse_video_source("rtsp://camera.example/live"),
            config=config,
            capture_factory=factory_from([failed, recovered]),
        )

        frame = reader.read()

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.frame_id, 0)
        self.assertAlmostEqual(frame.source_pts_seconds or 0.0, 0.123)
        stats = reader.stats()
        self.assertEqual(stats.open_count, 2)
        self.assertEqual(stats.read_failures, 1)
        self.assertEqual(stats.reconnect_attempts, 1)
        self.assertTrue(failed.released)

    def test_live_source_raises_after_reconnect_budget_is_exhausted(self) -> None:
        reader = OpenCVFrameReader(
            parse_video_source("srt://127.0.0.1:9000"),
            config=OpenCVReaderConfig(max_reconnect_attempts=0),
            capture_factory=factory_from([FakeCapture([(False, None)])]),
        )
        with self.assertRaises(StreamUnavailable):
            reader.read()

    def test_repeated_malformed_live_frames_trigger_reconnect(self) -> None:
        malformed = FakeCapture([(True, None), (True, None)])
        recovered = FakeCapture([(True, FakeFrame())])
        reader = OpenCVFrameReader(
            parse_video_source("rtsp://camera.example/live"),
            config=OpenCVReaderConfig(
                bad_frame_reconnect_threshold=2,
                max_reconnect_attempts=1,
                backoff=ExponentialBackoff(initial_seconds=0.0, maximum_seconds=0.0),
            ),
            capture_factory=factory_from([malformed, recovered]),
        )

        frame = reader.read()

        self.assertIsNotNone(frame)
        self.assertEqual(reader.stats().bad_frames, 2)
        self.assertEqual(reader.stats().reconnect_attempts, 1)

    def test_live_decoder_exception_triggers_reconnect(self) -> None:
        reader = OpenCVFrameReader(
            parse_video_source("rtsp://camera.example/live"),
            config=OpenCVReaderConfig(
                max_reconnect_attempts=1,
                backoff=ExponentialBackoff(initial_seconds=0.0, maximum_seconds=0.0),
            ),
            capture_factory=factory_from([RaisingCapture(), FakeCapture([(True, FakeFrame())])]),
        )

        self.assertIsNotNone(reader.read())
        self.assertEqual(reader.stats().read_failures, 1)
        self.assertEqual(reader.stats().reconnect_attempts, 1)

    def test_backoff_is_bounded(self) -> None:
        backoff = ExponentialBackoff(initial_seconds=0.5, maximum_seconds=2.0, multiplier=2.0)
        self.assertEqual([backoff.delay(index) for index in range(1, 6)], [0.5, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual(backoff.delay(100_000), 2.0)


if __name__ == "__main__":
    unittest.main()
