from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from seahunter.data import DatasetManifest, DatasetSplit, assert_no_test_leakage, audit_dataset, load_dataset_manifest
from seahunter.tools.data_audit import main


class DatasetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = Path(__file__).resolve().parents[1] / "configs/data/m5_manifest.example.json"

    def test_example_manifest_covers_weather_sizes_hard_negatives_and_version(self) -> None:
        report = audit_dataset(load_dataset_manifest(self.manifest_path))

        self.assertTrue(report.passes_blocking_gates)
        self.assertEqual(report.sample_count, 7)
        self.assertEqual(report.split_samples, {"test": 5, "train": 1, "validation": 1})
        self.assertEqual(report.missing_test_weather, ())
        self.assertEqual(set(report.pixel_size_buckets), {"tiny", "small", "medium", "large"})
        self.assertTrue(all(report.pixel_size_buckets.values()))
        self.assertEqual(set(report.hard_negative_counts), {"bird", "foam", "reflection", "wake", "wave"})
        assert_no_test_leakage(report)

    def test_video_or_voyage_cross_split_is_blocking(self) -> None:
        manifest = load_dataset_manifest(self.manifest_path)
        first, second, *remaining = manifest.samples
        leaked = replace(second, video_id=first.video_id, split=DatasetSplit.TEST)
        report = audit_dataset(replace(manifest, samples=(first, leaked, *remaining)))

        self.assertFalse(report.passes_blocking_gates)
        self.assertTrue(any(item.group_kind == "video" for item in report.leakage))
        with self.assertRaisesRegex(ValueError, "split leakage"):
            assert_no_test_leakage(report)

    def test_duplicate_sample_identity_is_rejected(self) -> None:
        manifest = load_dataset_manifest(self.manifest_path)
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            DatasetManifest(
                dataset_id=manifest.dataset_id,
                version=manifest.version,
                created_at=manifest.created_at,
                samples=(manifest.samples[0], manifest.samples[0]),
            )

    def test_cli_writes_machine_readable_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            self.assertEqual(main([str(self.manifest_path), "--output", str(output), "--require-gates"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passes_blocking_gates"])


if __name__ == "__main__":
    unittest.main()
