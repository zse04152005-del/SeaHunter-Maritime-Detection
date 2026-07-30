from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime
from pathlib import Path

RUN_MODEL_TESTS = os.getenv("SEAHUNTER_RUN_MODEL_TESTS") == "1"


@unittest.skipUnless(RUN_MODEL_TESTS, "set SEAHUNTER_RUN_MODEL_TESTS=1 to run PyTorch baseline tests")
class LegacyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np
        import torch

        from seahunter.runtime import activate_legacy_ultralytics

        cls.np = np
        cls.torch = torch
        cls.root = Path(__file__).resolve().parents[1]
        cls.ultralytics = activate_legacy_ultralytics(cls.root)

    def test_custom_modules_forward_backward_and_validation(self) -> None:
        from ultralytics.nn.modules import EMA, MultiSEAM, SPDConv

        tensor = self.torch.randn(1, 64, 32, 32, requires_grad=True)
        spd = SPDConv(64, 128)
        ema = EMA(128, 128, factor=32)
        seam = MultiSEAM(128, 128)
        output = seam(ema(spd(tensor)))
        self.assertEqual(tuple(output.shape), (1, 128, 16, 16))
        output.mean().backward()
        self.assertIsNotNone(tensor.grad)

        with self.assertRaisesRegex(ValueError, "even spatial dimensions"):
            spd(self.torch.randn(1, 64, 31, 32))
        with self.assertRaisesRegex(ValueError, "divisible"):
            EMA(100, 100, factor=32)

    def test_nwd_similarity_and_parameters(self) -> None:
        from ultralytics.utils.loss import BboxLoss, wasserstein_loss

        target = self.torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        same = wasserstein_loss(target, target)
        shifted = wasserstein_loss(target, target + 10.0)
        self.assertGreater(float(same), float(shifted))
        self.assertAlmostEqual(float(same), 1.0, places=3)
        self.assertEqual(BboxLoss(nwd_weight=0.0).nwd_weight, 0.0)
        with self.assertRaises(ValueError):
            BboxLoss(nwd_weight=1.1)

    def test_weight_load_and_blank_frame_inference(self) -> None:
        model = self.ultralytics.YOLO(str(self.root / "weights" / "seahunter_best.pt"))
        image = self.np.zeros((256, 256, 3), dtype=self.np.uint8)
        results = model.predict(source=image, imgsz=256, device="cpu", verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(tuple(results[0].orig_shape), (256, 256))

    def test_framework_neutral_detector_adapter(self) -> None:
        from seahunter.runtime import UltralyticsDetector
        from seahunter.schemas import FramePacket

        detector = UltralyticsDetector(
            self.root / "weights/seahunter_best.pt",
            repo_root=self.root,
            imgsz=256,
            device="cpu",
        )
        frame = FramePacket(
            source_id="smoke",
            frame_id=0,
            captured_at=datetime.now(UTC),
            width=256,
            height=256,
            payload=self.np.zeros((256, 256, 3), dtype=self.np.uint8),
        )
        detections = detector.infer(frame)
        self.assertIsInstance(detections, list)
        self.assertTrue(detector.detector_id.startswith("ultralytics-8.3.234:"))


if __name__ == "__main__":
    unittest.main()
