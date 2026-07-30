"""OpenCV/FFmpeg video decoding with timestamps and live-source recovery."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from threading import Event, RLock
from time import monotonic
from typing import Any, Literal, Protocol

from seahunter.schemas import FramePacket, SourceKind

from .sources import VideoSource

DecoderBackend = Literal["auto", "ffmpeg", "gstreamer"]


class VideoReaderError(RuntimeError):
    """Base error raised by video reader implementations."""


class VideoSourceOpenError(VideoReaderError):
    """Raised when a source cannot be opened."""


class StreamUnavailable(VideoReaderError):
    """Raised when a live source exhausts its configured reconnect budget."""


class ReaderClosed(VideoReaderError):
    """Raised when reading from a permanently closed reader."""


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Bounded exponential delay used between live-stream reconnect attempts."""

    initial_seconds: float = 0.25
    maximum_seconds: float = 8.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.initial_seconds < 0:
            raise ValueError("initial_seconds must be non-negative")
        if self.maximum_seconds < self.initial_seconds:
            raise ValueError("maximum_seconds must be at least initial_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")

    def delay(self, attempt: int) -> float:
        """Return the delay for a one-based retry attempt."""

        if attempt <= 0:
            raise ValueError("attempt must be positive")
        try:
            scaled = self.initial_seconds * self.multiplier ** (attempt - 1)
        except OverflowError:
            return self.maximum_seconds
        return min(self.maximum_seconds, scaled)


@dataclass(frozen=True, slots=True)
class OpenCVReaderConfig:
    """Runtime controls for OpenCV's FFmpeg/GStreamer capture backend."""

    backend: DecoderBackend = "ffmpeg"
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 2.0
    prefer_hardware_decode: bool = True
    fallback_fps: float = 30.0
    bad_frame_reconnect_threshold: int = 3
    max_reconnect_attempts: int | None = None
    backoff: ExponentialBackoff = ExponentialBackoff()

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "ffmpeg", "gstreamer"}:
            raise ValueError(f"unsupported decoder backend: {self.backend}")
        if self.open_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("open and read timeouts must be positive")
        if self.fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive")
        if self.bad_frame_reconnect_threshold <= 0:
            raise ValueError("bad_frame_reconnect_threshold must be positive")
        if self.max_reconnect_attempts is not None and self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be non-negative or None")


@dataclass(frozen=True, slots=True)
class ReaderStats:
    """Monitoring snapshot for a video reader."""

    source_id: str
    opened: bool
    open_count: int
    open_failures: int
    frames_decoded: int
    read_failures: int
    bad_frames: int
    reconnect_attempts: int
    last_error: str | None


class CaptureBackend(Protocol):
    """Small testable wrapper around a concrete decoder capture."""

    def is_opened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def position_ms(self) -> float: ...

    def fps(self) -> float: ...

    def release(self) -> None: ...


class CaptureFactory(Protocol):
    """Factory contract allowing decoder behavior to be mocked in tests."""

    def __call__(self, source: VideoSource, config: OpenCVReaderConfig) -> CaptureBackend: ...


class _OpenCVCapture:
    def __init__(self, capture: Any, cv2_module: Any) -> None:
        self._capture = capture
        self._cv2 = cv2_module

    def is_opened(self) -> bool:
        return bool(self._capture.isOpened())

    def read(self) -> tuple[bool, Any]:
        success, frame = self._capture.read()
        return bool(success), frame

    def position_ms(self) -> float:
        return float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))

    def fps(self) -> float:
        return float(self._capture.get(self._cv2.CAP_PROP_FPS))

    def release(self) -> None:
        self._capture.release()


def _opencv_capture_factory(source: VideoSource, config: OpenCVReaderConfig) -> CaptureBackend:
    """Create an OpenCV capture while preferring FFmpeg and hardware decode."""

    cv2: Any = importlib.import_module("cv2")
    target: str | int = int(source.location) if source.kind is SourceKind.DEVICE else source.location

    backend_map = {
        "auto": cv2.CAP_ANY,
        "ffmpeg": cv2.CAP_FFMPEG,
        "gstreamer": cv2.CAP_GSTREAMER,
    }
    api_preference = cv2.CAP_ANY if source.kind is SourceKind.DEVICE else backend_map[config.backend]
    timeout_params: list[int] = []
    if source.is_network:
        timeout_params.extend(
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                round(config.open_timeout_seconds * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                round(config.read_timeout_seconds * 1000),
            ]
        )

    hardware_params: list[int] = []
    if config.prefer_hardware_decode and source.kind is not SourceKind.DEVICE:
        acceleration_property = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        acceleration_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
        if acceleration_property is not None and acceleration_any is not None:
            hardware_params.extend([int(acceleration_property), int(acceleration_any)])

    def create(params: list[int]) -> Any:
        if params:
            return cv2.VideoCapture(target, api_preference, params)
        if api_preference == cv2.CAP_ANY:
            return cv2.VideoCapture(target)
        return cv2.VideoCapture(target, api_preference)

    capture = create(timeout_params + hardware_params)
    if not capture.isOpened() and hardware_params:
        capture.release()
        capture = create(timeout_params)
    return _OpenCVCapture(capture, cv2)


class OpenCVFrameReader:
    """Decode file/device/RTSP/SRT/HTTP sources one frame at a time.

    Files use their media PTS on a deterministic UTC epoch timeline. Live sources
    use the wall-clock time immediately before decode as the best available capture
    estimate. RTSP/SRT/device failures trigger bounded or unbounded reconnection.
    """

    _MEDIA_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def __init__(
        self,
        source: VideoSource,
        *,
        source_id: str | None = None,
        config: OpenCVReaderConfig | None = None,
        capture_factory: CaptureFactory = _opencv_capture_factory,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic_clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        normalized_id = (source_id or source.suggested_id()).strip()
        if not normalized_id:
            raise ValueError("source_id must not be empty")
        self.source = source
        self.source_id = normalized_id
        self.config = config or OpenCVReaderConfig()
        self._capture_factory = capture_factory
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._closed_event = Event()
        self._capture: CaptureBackend | None = None
        self._lock = RLock()
        self._closed = False
        self._next_frame_id = 0
        self._last_file_pts: float | None = None
        self._open_count = 0
        self._open_failures = 0
        self._frames_decoded = 0
        self._read_failures = 0
        self._bad_frames = 0
        self._reconnect_attempts = 0
        self._last_error: str | None = None

    def open(self) -> None:
        """Open the source once; retry policy is applied by :meth:`read`."""

        with self._lock:
            if self._closed:
                raise ReaderClosed("reader is closed")
            if self._capture is not None and self._capture.is_opened():
                return

        if self.source.kind is SourceKind.FILE:
            path = Path(self.source.location)
            if not path.is_file():
                raise FileNotFoundError(path)

        try:
            capture = self._capture_factory(self.source, self.config)
        except Exception as exc:
            self._record_open_failure(exc)
            raise VideoSourceOpenError(f"failed to create capture for {self.source_id}") from exc

        if not capture.is_opened():
            capture.release()
            error = VideoSourceOpenError(f"failed to open video source {self.source_id}")
            self._record_open_failure(error)
            raise error

        with self._lock:
            if self._closed:
                capture.release()
                raise ReaderClosed("reader was closed while opening")
            self._capture = capture
            self._open_count += 1
            self._last_error = None

    def read(self) -> FramePacket | None:
        """Read the next valid frame, reconnecting live sources when required."""

        retry = 0
        consecutive_bad_frames = 0
        while True:
            try:
                self.open()
            except (FileNotFoundError, ReaderClosed):
                raise
            except VideoSourceOpenError as exc:
                retry = self._retry_or_raise(retry, exc)
                continue

            captured_live_at = self._aware_clock_value()
            decode_started = self._monotonic_clock()
            try:
                with self._lock:
                    if self._closed:
                        raise ReaderClosed("reader is closed")
                    capture = self._capture
                    if capture is None:
                        continue
                    success, payload = capture.read()
                    position_ms = capture.position_ms() if success else float("nan")
                    fps = capture.fps() if success else float("nan")
            except ReaderClosed:
                raise
            except Exception as exc:
                with self._lock:
                    self._read_failures += 1
                    self._last_error = str(exc)
                self._release_capture()
                error = VideoReaderError(f"decoder raised while reading {self.source_id}")
                if not self.source.is_live:
                    raise error from exc
                retry = self._retry_or_raise(retry, error)
                continue
            decoded_at = self._aware_clock_value()
            decode_duration_ms = max(0.0, (self._monotonic_clock() - decode_started) * 1000.0)

            if success:
                dimensions = self._frame_dimensions(payload)
                if dimensions is None:
                    consecutive_bad_frames += 1
                    with self._lock:
                        self._bad_frames += 1
                        self._last_error = "decoder returned an empty or malformed frame"
                    if self.source.is_live and consecutive_bad_frames >= self.config.bad_frame_reconnect_threshold:
                        self._release_capture()
                        error = StreamUnavailable(f"too many malformed frames: {self.source_id}")
                        retry = self._retry_or_raise(retry, error)
                        consecutive_bad_frames = 0
                    continue

                height, width = dimensions
                source_pts = self._resolve_pts(position_ms, fps)
                if self.source.kind is SourceKind.FILE:
                    if source_pts is None:
                        raise RuntimeError("file PTS resolution unexpectedly returned None")
                    captured_at = self._MEDIA_EPOCH + timedelta(seconds=source_pts)
                else:
                    captured_at = captured_live_at
                with self._lock:
                    frame_id = self._next_frame_id
                    self._next_frame_id += 1
                    self._frames_decoded += 1
                    self._last_error = None
                return FramePacket(
                    source_id=self.source_id,
                    frame_id=frame_id,
                    captured_at=captured_at,
                    decoded_at=decoded_at,
                    source_pts_seconds=source_pts,
                    decode_duration_ms=decode_duration_ms,
                    width=width,
                    height=height,
                    payload=payload,
                )

            with self._lock:
                self._read_failures += 1
                self._last_error = "decoder read failed"
            if not self.source.is_live:
                return None

            self._release_capture()
            retry = self._retry_or_raise(retry, StreamUnavailable(f"stream read failed: {self.source_id}"))

    def close(self) -> None:
        """Release decoder resources. Closing is idempotent and permanent."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._closed_event.set()
        self._release_capture()

    def stats(self) -> ReaderStats:
        """Return a consistent monitoring snapshot."""

        with self._lock:
            return ReaderStats(
                source_id=self.source_id,
                opened=self._capture is not None and self._capture.is_opened(),
                open_count=self._open_count,
                open_failures=self._open_failures,
                frames_decoded=self._frames_decoded,
                read_failures=self._read_failures,
                bad_frames=self._bad_frames,
                reconnect_attempts=self._reconnect_attempts,
                last_error=self._last_error,
            )

    def __iter__(self) -> Iterator[FramePacket]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> OpenCVFrameReader:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _record_open_failure(self, error: BaseException) -> None:
        with self._lock:
            self._open_failures += 1
            self._last_error = str(error)

    def _retry_or_raise(self, completed_attempts: int, error: VideoReaderError) -> int:
        if not self.source.is_live:
            raise error
        maximum = self.config.max_reconnect_attempts
        if maximum is not None and completed_attempts >= maximum:
            raise StreamUnavailable(
                f"reconnect budget exhausted for {self.source_id} after {completed_attempts} attempts"
            ) from error

        attempt = completed_attempts + 1
        with self._lock:
            self._reconnect_attempts += 1
            self._last_error = str(error)
        delay = self.config.backoff.delay(attempt)
        if self._sleeper is None:
            if self._closed_event.wait(delay):
                raise ReaderClosed("reader closed during reconnect backoff")
        else:
            self._sleeper(delay)
        return attempt

    def _release_capture(self) -> None:
        with self._lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()

    def _resolve_pts(self, position_ms: float, fps: float) -> float | None:
        candidate = position_ms / 1000.0 if isfinite(position_ms) and position_ms >= 0 else None
        valid_fps = fps if isfinite(fps) and fps > 0 else self.config.fallback_fps
        if self.source.kind is not SourceKind.FILE:
            return candidate

        fallback = self._next_frame_id / valid_fps
        if candidate is None or (self._last_file_pts is not None and candidate <= self._last_file_pts):
            candidate = fallback if self._last_file_pts is None else self._last_file_pts + 1.0 / valid_fps
        self._last_file_pts = candidate
        return candidate

    def _aware_clock_value(self) -> datetime:
        value = self._utc_clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("utc_clock must return a timezone-aware datetime")
        return value

    @staticmethod
    def _frame_dimensions(payload: Any) -> tuple[int, int] | None:
        shape = getattr(payload, "shape", None)
        if shape is None or len(shape) < 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return None
        return height, width
