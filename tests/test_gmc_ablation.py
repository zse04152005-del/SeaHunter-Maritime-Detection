from __future__ import annotations

import importlib.util
import unittest

from seahunter.evaluation.gmc_ablation import run_synthetic_gmc_ablation

HAS_VIDEO_STACK = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_VIDEO_STACK, "OpenCV and NumPy are required for the GMC ablation")
class GMCAblationTests(unittest.TestCase):
    def test_visual_gmc_removes_synthetic_pan_identity_switches(self) -> None:
        report = run_synthetic_gmc_ablation()
        baseline = report["baseline_no_gmc"]
        compensated = report["botsort_with_gmc"]
        improvement = report["improvement"]
        assert isinstance(baseline, dict)
        assert isinstance(compensated, dict)
        assert isinstance(improvement, dict)

        self.assertEqual(baseline["unique_track_ids"], [1, 2, 3, 4])
        self.assertEqual(compensated["unique_track_ids"], [1])
        self.assertEqual(improvement["baseline_id_switches"], 3)
        self.assertEqual(improvement["compensated_id_switches"], 0)
        self.assertEqual(improvement["id_switch_reduction"], 3)
        motion = compensated["motion"]
        assert isinstance(motion, dict)
        self.assertEqual(motion["frames_applied"], 3)
        self.assertFalse(report["dataset_claim"])


if __name__ == "__main__":
    unittest.main()
