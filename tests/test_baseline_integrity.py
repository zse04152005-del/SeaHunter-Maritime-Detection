from __future__ import annotations

import unittest
from pathlib import Path

from seahunter.runtime import legacy_source_path
from seahunter.tools.baseline_audit import EXPECTED_WEIGHT_SHA256, audit_baseline


class BaselineIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_static_baseline_audit(self) -> None:
        result = audit_baseline(self.root)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["weight"]["sha256"], EXPECTED_WEIGHT_SHA256)

    def test_legacy_source_is_local(self) -> None:
        source = legacy_source_path(self.root)
        self.assertEqual(source, self.root / "yolo_source")


if __name__ == "__main__":
    unittest.main()
