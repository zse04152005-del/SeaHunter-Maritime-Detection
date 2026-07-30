from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone

from seahunter.schemas import FramePacket
from seahunter.tracking import SparseOpticalFlowConfig, SparseOpticalFlowGMC

HAS_VIDEO_STACK = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


def frame(frame_id: int, payload: object | None) -> FramePacket:
    return FramePacket(
        source_id="synthetic-pan",
        frame_id=frame_id,
        captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        width=320,
        height=240,
        payload=payload,
    )


@unittest.skipUnless(HAS_VIDEO_STACK, "OpenCV and NumPy are required for visual GMC")
class SparseOpticalFlowGMCTests(unittest.TestCase):
    def test_recovers_synthetic_camera_translation(self) -> None:
        import cv2
        import numpy as np

        rng = np.random.default_rng(20260730)
        previous = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
        expected_x, expected_y = 24.0, 10.0
        current = cv2.warpAffine(
            previous,
            np.asarray(((1.0, 0.0, expected_x), (0.0, 1.0, expected_y)), dtype=float),
            (320, 240),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        estimator = SparseOpticalFlowGMC(SparseOpticalFlowConfig(minimum_inliers=20))
        estimate = estimator.estimate(frame(0, previous), frame(1, current))

        self.assertTrue(estimate.applied)
        self.assertIsNone(estimate.fallback_reason)
        self.assertGreaterEqual(estimate.inliers, 20)
        self.assertGreaterEqual(estimate.quality, 0.35)
        self.assertAlmostEqual(estimate.translation_xy[0], expected_x, delta=1.0)
        self.assertAlmostEqual(estimate.translation_xy[1], expected_y, delta=1.0)

    def test_missing_payload_falls_back_to_identity(self) -> None:
        estimator = SparseOpticalFlowGMC()
        estimate = estimator.estimate(frame(0, None), frame(1, None))

        self.assertFalse(estimate.applied)
        self.assertEqual(estimate.fallback_reason, "missing_or_invalid_payload")
        self.assertEqual(estimate.translation_xy, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
