from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.experiments import load_detector_experiment_suite
from seahunter.tools.experiment_runner import main


class DetectorExperimentPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = cls.root / "configs/experiments/m0_detector_nwd_ablation.json"

    def test_real_nwd_ablation_expands_variants_and_three_seeds(self) -> None:
        suite = load_detector_experiment_suite(self.config, repo_root=self.root)
        self.assertEqual(len(suite.runs), 6)
        self.assertEqual({run.variant_id for run in suite.runs}, {"ciou-only", "ciou-nwd-50"})
        self.assertEqual({run.seed for run in suite.runs}, {0, 1, 2})
        self.assertEqual({run.nwd_weight for run in suite.runs}, {0.0, 0.5})
        self.assertEqual(len(suite.source_sha256), 64)

    def test_dry_run_prints_selected_plan_without_importing_training_stack(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(["--config", str(self.config), "--variant", "ciou-only", "--dry-run"]),
                0,
            )
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["run_count"], 6)
        self.assertEqual(plan["selected_run_count"], 3)
        self.assertEqual({run["variant_id"] for run in plan["selected_runs"]}, {"ciou-only"})

    def test_duplicate_variant_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs/models").mkdir(parents=True)
            (root / "configs/data").mkdir(parents=True)
            (root / "weights").mkdir()
            (root / "configs/models/model.yaml").touch()
            (root / "configs/data/data.yaml").touch()
            (root / "weights/init.pt").touch()
            payload = {
                "schema_version": 1,
                "suite_id": "duplicate-suite",
                "description": "duplicate variant validation",
                "defaults": {
                    "model": "configs/models/model.yaml",
                    "data": "configs/data/data.yaml",
                    "initialization_weights": "weights/init.pt",
                    "project": "runs/test",
                    "epochs": 1,
                    "image_size": 64,
                    "batch_size": 1,
                    "workers": 0,
                    "nwd_constant": 12.8,
                },
                "variants": [
                    {"id": "same", "description": "first", "nwd_weight": 0.0},
                    {"id": "same", "description": "second", "nwd_weight": 0.5},
                ],
                "seeds": [0],
            }
            config = root / "configs/duplicate.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate variant"):
                load_detector_experiment_suite(config, repo_root=root)


if __name__ == "__main__":
    unittest.main()
