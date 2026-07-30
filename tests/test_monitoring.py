from __future__ import annotations

import unittest

from seahunter.runtime import ResourceSample, RuntimeResourceMonitor


class SequenceProvider:
    def __init__(self, samples: list[ResourceSample]) -> None:
        self.samples = list(samples)

    def sample(self) -> ResourceSample:
        return self.samples.pop(0)


class FailingProvider:
    def sample(self) -> ResourceSample:
        raise RuntimeError("metrics unavailable")


class MonitoringTests(unittest.TestCase):
    def test_periodic_monitor_retains_latest_values_and_peaks(self) -> None:
        times = iter([0.0, 0.5, 1.5])
        provider = SequenceProvider(
            [
                ResourceSample(100.0, 10.0, 50.0, 200.0, 40.0, 60.0, 25.0),
                ResourceSample(125.0, 20.0, 55.0, 180.0, 70.0, 65.0, 30.0),
            ]
        )
        monitor = RuntimeResourceMonitor(
            provider=provider,
            interval_seconds=1.0,
            monotonic_clock=lambda: next(times),
        )

        self.assertIsNotNone(monitor.observe())
        self.assertIsNone(monitor.observe())
        self.assertIsNotNone(monitor.observe())
        summary = monitor.summary()

        self.assertEqual(summary.samples, 2)
        self.assertEqual(summary.peak_process_rss_mb, 125.0)
        self.assertEqual(summary.peak_gpu_memory_used_mb, 200.0)
        self.assertEqual(summary.peak_gpu_utilization_percent, 70.0)
        self.assertEqual(summary.latest_process_cpu_percent, 20.0)

    def test_sampling_failure_is_reported_without_crashing(self) -> None:
        monitor = RuntimeResourceMonitor(provider=FailingProvider(), interval_seconds=0.0)
        self.assertIsNone(monitor.observe())
        summary = monitor.summary()
        self.assertEqual(summary.sampling_failures, 1)
        self.assertIn("metrics unavailable", summary.last_error or "")


if __name__ == "__main__":
    unittest.main()
