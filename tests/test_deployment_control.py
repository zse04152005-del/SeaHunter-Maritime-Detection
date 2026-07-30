from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from seahunter.deployment import (
    EdgeDegradationConfig,
    EdgeDegradationController,
    FaultResult,
    FaultScenario,
    ReleaseManager,
    ReleasePhase,
    RuntimeHealth,
    RuntimeMode,
    SoakAccumulator,
    SoakSample,
    validate_fault_matrix,
)


class DeploymentControlTests(unittest.TestCase):
    def test_thermal_memory_latency_and_timeout_degrade_then_recover_with_hysteresis(self) -> None:
        controller = EdgeDegradationController(EdgeDegradationConfig(recovery_samples=2))
        reduced = controller.update(RuntimeHealth(82.0, 0.5, 30.0))
        lower = controller.update(RuntimeHealth(82.0, 0.95, 70.0))
        stopped = controller.update(RuntimeHealth(92.0, 0.95, 70.0, timeout_streak=3))
        recovering = controller.update(RuntimeHealth(60.0, 0.3, 20.0))
        recovered = controller.update(RuntimeHealth(60.0, 0.3, 20.0))

        self.assertEqual(reduced.mode, RuntimeMode.REDUCED_ROI)
        self.assertEqual(lower.mode, RuntimeMode.LOWER_RESOLUTION)
        self.assertEqual(stopped.mode, RuntimeMode.SAFE_STOP)
        self.assertEqual(recovering.reasons, ("recovery_hysteresis",))
        self.assertEqual(recovered.mode, RuntimeMode.NORMAL)

    def test_canary_failure_rolls_back_and_success_persists_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "release.json"
            manager = ReleaseManager(state_path, initial_version="v1", required_healthy_samples=2)
            manager.stage("v2")
            manager.begin_canary()
            rolled_back = manager.report_canary_health(False)
            self.assertEqual(rolled_back.phase, ReleasePhase.ROLLED_BACK)
            self.assertEqual(rolled_back.current_version, "v1")

            manager.stage("v2")
            manager.begin_canary()
            manager.report_canary_health(True)
            promoted = manager.report_canary_health(True)
            self.assertEqual(promoted.phase, ReleasePhase.STABLE)
            self.assertEqual(promoted.current_version, "v2")
            self.assertEqual(promoted.previous_version, "v1")
            restarted = ReleaseManager(state_path, initial_version="ignored")
            self.assertEqual(restarted.state, promoted)

    def test_soak_accumulator_is_constant_state_and_enforces_duration_memory_fatal_gates(self) -> None:
        accumulator = SoakAccumulator()
        accumulator.add(SoakSample(0.0, 100.0, 200.0, 20.0, 0))
        accumulator.add(SoakSample(8 * 3600.0, 100.8, 220.0, 25.0, 800_000, recoverable_errors=2))
        report = accumulator.report()

        self.assertAlmostEqual(report.memory_slope_mb_per_hour, 0.1)
        self.assertTrue(report.passes(required_hours=8.0, maximum_memory_slope_mb_per_hour=1.0))
        self.assertFalse(report.passes(required_hours=72.0, maximum_memory_slope_mb_per_hour=1.0))

    def test_fault_matrix_requires_every_recoverable_scenario(self) -> None:
        results = tuple(FaultResult(scenario, True, 1.0, f"evidence:{scenario.value}") for scenario in FaultScenario)
        validate_fault_matrix(results)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_fault_matrix(results[:-1])
        failed = (*results[:-1], FaultResult(results[-1].scenario, False, None, "evidence:failed"))
        with self.assertRaisesRegex(ValueError, "recovery failed"):
            validate_fault_matrix(failed)


if __name__ == "__main__":
    unittest.main()
