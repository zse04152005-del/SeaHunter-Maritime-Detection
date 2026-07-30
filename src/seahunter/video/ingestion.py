"""Threaded bounded video ingestion for freshness-first real-time pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Protocol

from seahunter.schemas import FramePacket

from .buffer import BufferClosed, BufferStats, LatestItemBuffer
from .sources import VideoSource


class FrameReader(Protocol):
    """Minimal reader contract consumed by replay and edge services."""

    source: VideoSource

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


class IngestionFailed(RuntimeError):
    """Raised in the consumer when the producer thread failed."""


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """Monitoring snapshot for the threaded ingestion worker."""

    running: bool
    produced: int
    failure: str | None
    buffer: BufferStats


class FrameIngestWorker:
    """Continuously decode into a bounded drop-oldest buffer."""

    def __init__(self, reader: FrameReader, *, capacity: int = 2, thread_name: str = "seahunter-ingest") -> None:
        self.reader = reader
        self.buffer = LatestItemBuffer[FramePacket](capacity)
        self._thread_name = thread_name
        self._thread: Thread | None = None
        self._stop = Event()
        self._state_lock = Lock()
        self._produced = 0
        self._failure: BaseException | None = None

    def start(self) -> None:
        """Start the producer exactly once."""

        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("ingestion worker has already been started")
            self._thread = Thread(target=self._run, name=self._thread_name, daemon=True)
            self._thread.start()

    def get(self, timeout: float | None = None) -> FramePacket:
        """Return the next retained frame or surface the producer failure."""

        if self._thread is None:
            raise RuntimeError("ingestion worker has not been started")
        try:
            return self.buffer.get(timeout)
        except BufferClosed as exc:
            with self._state_lock:
                failure = self._failure
            if failure is not None:
                raise IngestionFailed("video ingestion failed") from failure
            raise exc

    def stop(self) -> None:
        """Request shutdown, release the reader, and wake consumers."""

        self._stop.set()
        try:
            self.reader.close()
        finally:
            self.buffer.close()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for producer termination and return whether it stopped."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def stats(self) -> IngestionStats:
        """Return worker and queue monitoring counters."""

        with self._state_lock:
            thread = self._thread
            produced = self._produced
            failure = None if self._failure is None else repr(self._failure)
        return IngestionStats(
            running=thread is not None and thread.is_alive(),
            produced=produced,
            failure=failure,
            buffer=self.buffer.stats(),
        )

    def __enter__(self) -> FrameIngestWorker:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
        self.join(timeout=5.0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.reader.read()
                if frame is None:
                    return
                try:
                    self.buffer.put(frame)
                except BufferClosed:
                    return
                with self._state_lock:
                    self._produced += 1
        except BaseException as exc:
            if not self._stop.is_set():
                with self._state_lock:
                    self._failure = exc
        finally:
            try:
                self.reader.close()
            except BaseException as exc:
                if not self._stop.is_set():
                    with self._state_lock:
                        if self._failure is None:
                            self._failure = exc
            finally:
                self.buffer.close()
