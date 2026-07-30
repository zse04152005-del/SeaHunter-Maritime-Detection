from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seahunter.tools.acceptance_audit import main


class AcceptanceCLITests(unittest.TestCase):
    def test_template_generates_readiness_gaps_and_cannot_pass_required_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        template = root / "configs/acceptance/m8_submission.template.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "readiness.json"
            self.assertEqual(main([str(template), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertIn("locations", report["missing_coverage"])
            self.assertEqual(main([str(template), "--require-passed"]), 2)


if __name__ == "__main__":
    unittest.main()
