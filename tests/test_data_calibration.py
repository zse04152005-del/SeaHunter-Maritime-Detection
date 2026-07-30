from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from seahunter.data import (
    CalibrationSample,
    ThresholdSample,
    WeatherCondition,
    build_experiment_provenance,
    evaluate_weather_thresholds,
    fit_temperature,
    write_provenance,
)


class DataCalibrationTests(unittest.TestCase):
    def test_temperature_scaling_returns_finite_bounded_calibration(self) -> None:
        samples = (
            CalibrationSample(5.0, True, WeatherCondition.NORMAL),
            CalibrationSample(3.0, True, WeatherCondition.NORMAL),
            CalibrationSample(-4.0, False, WeatherCondition.FOG),
            CalibrationSample(-2.0, False, WeatherCondition.FOG),
        )
        calibration = fit_temperature(samples)

        self.assertGreater(calibration.temperature, 0.0)
        self.assertGreater(calibration.probability(5.0), calibration.probability(-4.0))
        self.assertGreaterEqual(calibration.expected_calibration_error, 0.0)
        self.assertLessEqual(calibration.expected_calibration_error, 1.0)

    def test_weather_thresholds_are_evaluated_independently(self) -> None:
        samples = (
            ThresholdSample(0.9, True, WeatherCondition.NORMAL),
            ThresholdSample(0.4, False, WeatherCondition.NORMAL),
            ThresholdSample(0.85, True, WeatherCondition.FOG),
            ThresholdSample(0.8, False, WeatherCondition.FOG),
            ThresholdSample(0.7, True, WeatherCondition.FOG),
        )
        results = evaluate_weather_thresholds(samples, minimum_precision=1.0)

        self.assertEqual({item.weather for item in results}, {WeatherCondition.NORMAL, WeatherCondition.FOG})
        fog = next(item for item in results if item.weather is WeatherCondition.FOG)
        self.assertEqual(fog.false_positives, 0)
        self.assertGreaterEqual(fog.threshold, 0.85)

    def test_experiment_provenance_hashes_config_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("seed: 0\n", encoding="utf-8")
            provenance = build_experiment_provenance(
                run_id="run-1",
                git_commit="abc123",
                data_version="dvc:deadbeef",
                config_path=config,
                seed=0,
                model_id="teacher-student",
            )
            output = root / "run.json"
            write_provenance(output, provenance)

            self.assertEqual(len(provenance.config_sha256), 64)
            self.assertIn(provenance.config_sha256, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
