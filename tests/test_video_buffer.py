from __future__ import annotations

import unittest

from seahunter.video import BufferClosed, LatestItemBuffer


class LatestItemBufferTests(unittest.TestCase):
    def test_full_buffer_drops_oldest_and_keeps_latest(self) -> None:
        buffer = LatestItemBuffer[int](capacity=2)
        buffer.put(1)
        buffer.put(2)
        buffer.put(3)

        self.assertEqual(buffer.get(), 2)
        self.assertEqual(buffer.get(), 3)
        self.assertEqual(buffer.stats().dropped, 1)
        self.assertEqual(buffer.stats().offered, 3)
        self.assertEqual(buffer.stats().delivered, 2)

    def test_close_allows_drain_then_raises(self) -> None:
        buffer = LatestItemBuffer[str](capacity=1)
        buffer.put("last")
        buffer.close()

        self.assertEqual(buffer.get(), "last")
        with self.assertRaises(BufferClosed):
            buffer.get()

    def test_put_after_close_raises(self) -> None:
        buffer = LatestItemBuffer[int](capacity=1)
        buffer.close()
        with self.assertRaises(BufferClosed):
            buffer.put(1)

    def test_empty_buffer_times_out(self) -> None:
        buffer = LatestItemBuffer[int](capacity=1)
        with self.assertRaises(TimeoutError):
            buffer.get(timeout=0.001)

    def test_invalid_capacity_and_timeout(self) -> None:
        with self.assertRaises(ValueError):
            LatestItemBuffer[int](capacity=0)

        buffer = LatestItemBuffer[int](capacity=1)
        with self.assertRaises(ValueError):
            buffer.get(timeout=-1.0)


if __name__ == "__main__":
    unittest.main()
