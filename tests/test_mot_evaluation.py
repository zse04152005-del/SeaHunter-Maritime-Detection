from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.evaluation import (
    MOTEvaluationConfig,
    build_mot_error_index,
    evaluate_mot_sequence,
    load_mot_file,
)

ROOT = Path(__file__).resolve().parents[1]
GT_PATH = ROOT / "tests/fixtures/mot/synthetic-maritime-gt.txt"
PREDICTION_PATH = ROOT / "tests/fixtures/mot/synthetic-maritime-pred.txt"


class MOTEvaluationTests(unittest.TestCase):
    def test_parser_rejects_duplicate_identity_in_one_frame(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.txt"
            path.write_text(
                "1,7,0,0,10,10,1,1,1\n1,7,20,20,10,10,1,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate track_id"):
                load_mot_file(path, kind="ground_truth")

    def test_error_index_locates_switch_fragment_miss_and_false_positive(self) -> None:
        ground_truth = load_mot_file(GT_PATH, kind="ground_truth")
        predictions = load_mot_file(PREDICTION_PATH, kind="prediction")
        index = build_mot_error_index(ground_truth, predictions)
        summary = index["summary"]

        self.assertIsInstance(summary, dict)
        assert isinstance(summary, dict)
        self.assertEqual(summary["matches"], 7)
        self.assertEqual(summary["false_negatives"], 1)
        self.assertEqual(summary["false_positives"], 1)
        self.assertEqual(summary["id_switches"], 1)
        self.assertEqual(summary["fragmentations"], 1)
        frames = index["frames"]
        self.assertIsInstance(frames, list)
        assert isinstance(frames, list)
        self.assertEqual([frame["frame_id"] for frame in frames], [3, 4])

    def test_trackeval_reports_standard_metrics_for_synthetic_sequence(self) -> None:
        ground_truth = load_mot_file(GT_PATH, kind="ground_truth")
        predictions = load_mot_file(PREDICTION_PATH, kind="prediction")
        with TemporaryDirectory() as directory:
            report = evaluate_mot_sequence(
                ground_truth,
                predictions,
                config=MOTEvaluationConfig(sequence_name="synthetic-maritime"),
                artifacts_dir=Path(directory),
            )

        metrics = report["metrics"]
        self.assertIsInstance(metrics, dict)
        assert isinstance(metrics, dict)
        self.assertEqual(metrics["IDSwitches"], 1)
        self.assertEqual(metrics["Fragmentations"], 1)
        self.assertEqual(metrics["FalsePositives"], 1)
        self.assertEqual(metrics["FalseNegatives"], 1)
        self.assertEqual(metrics["GroundTruthDetections"], 8)
        self.assertEqual(metrics["PredictionDetections"], 8)
        self.assertAlmostEqual(float(metrics["MOTA"]), 0.625)
        self.assertGreater(float(metrics["HOTA"]), 0.0)
        self.assertLess(float(metrics["HOTA"]), 1.0)
        self.assertGreater(float(metrics["IDF1"]), 0.0)
        self.assertLess(float(metrics["IDF1"]), 1.0)

    def test_configuration_rejects_unsafe_sequence_name(self) -> None:
        with self.assertRaises(ValueError):
            MOTEvaluationConfig(sequence_name="../flight")


if __name__ == "__main__":
    unittest.main()
