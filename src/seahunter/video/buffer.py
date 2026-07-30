"""Thread-safe bounded buffer with an explicit drop-oldest policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class BufferClosed(RuntimeError):
    """Raised when reading from or writing to a closed buffer."""


@dataclass(frozen=True, slots=True)
class BufferStats:
    """Immutable buffer counters for monitoring and alerting."""

    capacity: int
    size: int
    offered: int
    delivered: int
    dropped: int
    closed: bool


class LatestItemBuffer(Generic[T]):
    """A bounded producer/consumer buffer that drops the oldest item when full.

    Real-time perception must process recent frames instead of accumulating stale
    work. Producers therefore never block on a full buffer: the oldest queued item
    is discarded and the newest item is retained.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: deque[T] = deque()
        self._condition = Condition()
        self._offered = 0
        self._delivered = 0
        self._dropped = 0
        self._closed = False

    def put(self, item: T) -> None:
        """Offer an item without blocking, dropping the oldest queued item if full."""

        with self._condition:
            if self._closed:
                raise BufferClosed("cannot put into a closed buffer")
            self._offered += 1
            if len(self._items) == self._capacity:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
            self._condition.notify()

    def get(self, timeout: float | None = None) -> T:
        """Return the oldest retained item, waiting until data or closure.

        A closed buffer may still be drained. ``BufferClosed`` is raised only when
        the buffer is both closed and empty.
        """

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")

        with self._condition:
            deadline = None if timeout is None else monotonic() + timeout
            while not self._items and not self._closed:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for an item")
                self._condition.wait(remaining)

            if not self._items:
                raise BufferClosed("buffer is closed and empty")

            self._delivered += 1
            return self._items.popleft()

    def close(self) -> None:
        """Prevent further writes and wake all waiting consumers."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def stats(self) -> BufferStats:
        """Return a consistent monitoring snapshot."""

        with self._condition:
            return BufferStats(
                capacity=self._capacity,
                size=len(self._items),
                offered=self._offered,
                delivered=self._delivered,
                dropped=self._dropped,
                closed=self._closed,
            )

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)
