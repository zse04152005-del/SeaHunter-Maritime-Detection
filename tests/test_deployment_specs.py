from __future__ import annotations

import unittest

from seahunter.data import WeatherCondition
from seahunter.deployment import (
    BackendConsistencyResult,
    HighPerformanceRuntimeSpec,
    InferencePrecision,
    INT8CalibrationSet,
    ONNXExportSpec,
    RuntimeCapabilities,
    TensorRTBuildSpec,
)


class DeploymentSpecTests(unittest.TestCase):
    def test_static_and_dynamic_onnx_specs_are_explicit(self) -> None:
        static = ONNXExportSpec("model.static.onnx", 18, (1, 3, 1024, 1024))
        dynamic = ONNXExportSpec("model.dynamic.onnx", 18, (1, 3, 1024, 1024), dynamic_batch=True)

        self.assertEqual(static.dynamic_axes, {})
        self.assertEqual(dynamic.dynamic_axes["images"], {0: "batch"})

    def test_fp16_build_arguments_include_profiles_without_claiming_execution(self) -> None:
        spec = TensorRTBuildSpec(
            "model.onnx",
            "model.engine",
            InferencePrecision.FP16,
            (1, 3, 768, 768),
            (1, 3, 1024, 1024),
            (4, 3, 1024, 1024),
        )
        arguments = spec.trtexec_arguments()

        self.assertIn("--fp16", arguments)
        self.assertIn("--minShapes=images:1x3x768x768", arguments)
        self.assertIn("--maxShapes=images:4x3x1024x1024", arguments)

    def test_int8_requires_real_complete_weather_calibration(self) -> None:
        incomplete = INT8CalibrationSet(
            "dvc:calibration-v1",
            10,
            False,
            ((WeatherCondition.NORMAL, 10),),
        )
        with self.assertRaisesRegex(ValueError, "real maritime"):
            incomplete.validate_for_production(minimum_per_weather=2)

        complete = INT8CalibrationSet(
            "dvc:calibration-v2",
            50,
            True,
            tuple((weather, 10) for weather in WeatherCondition),
        )
        complete.validate_for_production(minimum_per_weather=10)
        self.assertEqual(
            TensorRTBuildSpec(
                "model.onnx",
                "model.int8.engine",
                InferencePrecision.INT8,
                (1, 3, 1024, 1024),
                (1, 3, 1024, 1024),
                (1, 3, 1024, 1024),
                calibration=complete,
            ).precision,
            InferencePrecision.INT8,
        )

    def test_runtime_capabilities_and_numeric_consistency_are_hard_gates(self) -> None:
        runtime = HighPerformanceRuntimeSpec()
        missing = runtime.missing_capabilities(RuntimeCapabilities(False, True, False, True))
        self.assertEqual(missing, ("nvdec", "preallocated_buffers"))
        consistency = BackendConsistencyResult("onnx", True, 1e-4, 1e-6, 1e-3, 1e-5)
        self.assertTrue(consistency.passed)
        self.assertFalse(BackendConsistencyResult("tensorrt", True, 0.1, 0.01, 0.001, 0.0001).passed)


if __name__ == "__main__":
    unittest.main()
