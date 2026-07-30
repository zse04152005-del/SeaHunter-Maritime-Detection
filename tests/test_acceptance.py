from __future__ import annotations

import unittest
from dataclasses import replace

from seahunter.acceptance import (
    ReviewApproval,
    ReviewDomain,
    SeaState,
    SeaTrialPlan,
    SeaTrialResult,
    TimeBand,
    audit_acceptance,
)
from seahunter.data import WeatherCondition


def plan() -> SeaTrialPlan:
    return SeaTrialPlan(
        plan_id="v1-trial",
        minimum_locations=2,
        required_time_bands=tuple(TimeBand),
        required_altitude_bands=("low", "medium", "high"),
        required_weather=tuple(WeatherCondition),
        required_sea_states=tuple(SeaState),
        minimum_blind_operator_trials=5,
        maximum_false_alarms_per_hour=1.0,
        maximum_missed_alerts=0,
        minimum_reacquisition_rate=0.8,
        maximum_distance_mae_m=20.0,
    )


def trials() -> tuple[SeaTrialResult, ...]:
    time_bands = tuple(TimeBand)
    altitudes = ("low", "medium", "high")
    sea_states = tuple(SeaState)
    return tuple(
        SeaTrialResult(
            trial_id=f"trial-{index}",
            location_id=f"location-{index % 2}",
            time_band=time_bands[index % len(time_bands)],
            altitude_band=altitudes[index % len(altitudes)],
            weather=weather,
            sea_state=sea_states[index % len(sea_states)],
            blind_operator=True,
            operator_id=f"operator-{index}",
            calibration_report=f"calibration-{index}.json",
            telemetry_sync_report=f"sync-{index}.json",
            zone_config_version="zones-v1",
            event_audit_reference=f"events-{index}.json",
            evidence_bundle_reference=f"evidence-{index}.zip",
            model_package_sha256="a" * 64,
            false_alarms_per_hour=0.5,
            missed_alerts=0,
            reacquisition_rate=0.9,
            distance_mae_m=10.0,
        )
        for index, weather in enumerate(WeatherCondition)
    )


def approvals() -> tuple[ReviewApproval, ...]:
    return tuple(
        ReviewApproval(domain, True, f"reviewer-{domain.value}", f"review-{domain.value}.pdf")
        for domain in ReviewDomain
    )


class AcceptanceTests(unittest.TestCase):
    def test_complete_coverage_metrics_evidence_and_reviews_pass(self) -> None:
        audit = audit_acceptance(plan(), trials(), approvals())
        self.assertTrue(audit.passed)
        self.assertEqual(audit.trial_count, 5)

    def test_missing_weather_blind_trial_and_review_are_explicit(self) -> None:
        audit = audit_acceptance(plan(), trials()[:-1], approvals()[:-1])
        self.assertFalse(audit.passed)
        self.assertIn("weather:high_wave", audit.missing_coverage)
        self.assertIn("blind_operator_trials", audit.missing_coverage)
        self.assertIn(ReviewDomain.OPERATIONS.value, audit.missing_reviews)

    def test_any_per_trial_metric_failure_blocks_acceptance(self) -> None:
        failed = (replace(trials()[0], missed_alerts=1, distance_mae_m=30.0), *trials()[1:])
        audit = audit_acceptance(plan(), failed, approvals())
        self.assertIn("trial-0:missed_alerts", audit.metric_failures)
        self.assertIn("trial-0:distance_mae_m", audit.metric_failures)

    def test_duplicate_trials_or_reviews_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial IDs"):
            audit_acceptance(plan(), (trials()[0], trials()[0]), approvals())
        with self.assertRaisesRegex(ValueError, "review domains"):
            audit_acceptance(plan(), trials(), (approvals()[0], approvals()[0]))


if __name__ == "__main__":
    unittest.main()
