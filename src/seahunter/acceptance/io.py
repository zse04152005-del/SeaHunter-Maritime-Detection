"""JSON submission loader for sea-trial readiness audits."""

from __future__ import annotations

import json
from pathlib import Path

from seahunter.data import WeatherCondition

from .models import ReviewApproval, ReviewDomain, SeaState, SeaTrialPlan, SeaTrialResult, TimeBand


def load_acceptance_submission(
    path: Path,
) -> tuple[SeaTrialPlan, tuple[SeaTrialResult, ...], tuple[ReviewApproval, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("acceptance submission must be an object")
    plan_payload = payload.get("plan")
    trials_payload = payload.get("trials")
    reviews_payload = payload.get("reviews")
    if (
        not isinstance(plan_payload, dict)
        or not isinstance(trials_payload, list)
        or not isinstance(reviews_payload, list)
    ):
        raise ValueError("acceptance submission requires plan, trials, and reviews")
    plan = SeaTrialPlan(
        plan_id=str(plan_payload["plan_id"]),
        minimum_locations=int(plan_payload["minimum_locations"]),
        required_time_bands=tuple(TimeBand(str(value)) for value in _list(plan_payload["required_time_bands"])),
        required_altitude_bands=tuple(str(value) for value in _list(plan_payload["required_altitude_bands"])),
        required_weather=tuple(WeatherCondition(str(value)) for value in _list(plan_payload["required_weather"])),
        required_sea_states=tuple(SeaState(str(value)) for value in _list(plan_payload["required_sea_states"])),
        minimum_blind_operator_trials=int(plan_payload["minimum_blind_operator_trials"]),
        maximum_false_alarms_per_hour=float(plan_payload["maximum_false_alarms_per_hour"]),
        maximum_missed_alerts=int(plan_payload["maximum_missed_alerts"]),
        minimum_reacquisition_rate=float(plan_payload["minimum_reacquisition_rate"]),
        maximum_distance_mae_m=float(plan_payload["maximum_distance_mae_m"]),
    )
    trials = tuple(_trial(value) for value in trials_payload)
    reviews = tuple(_review(value) for value in reviews_payload)
    return plan, trials, reviews


def _trial(value: object) -> SeaTrialResult:
    if not isinstance(value, dict):
        raise ValueError("each sea trial must be an object")
    return SeaTrialResult(
        trial_id=str(value["trial_id"]),
        location_id=str(value["location_id"]),
        time_band=TimeBand(str(value["time_band"])),
        altitude_band=str(value["altitude_band"]),
        weather=WeatherCondition(str(value["weather"])),
        sea_state=SeaState(str(value["sea_state"])),
        blind_operator=bool(value["blind_operator"]),
        operator_id=str(value["operator_id"]),
        calibration_report=str(value["calibration_report"]),
        telemetry_sync_report=str(value["telemetry_sync_report"]),
        zone_config_version=str(value["zone_config_version"]),
        event_audit_reference=str(value["event_audit_reference"]),
        evidence_bundle_reference=str(value["evidence_bundle_reference"]),
        model_package_sha256=str(value["model_package_sha256"]),
        false_alarms_per_hour=float(value["false_alarms_per_hour"]),
        missed_alerts=int(value["missed_alerts"]),
        reacquisition_rate=float(value["reacquisition_rate"]),
        distance_mae_m=float(value["distance_mae_m"]),
    )


def _review(value: object) -> ReviewApproval:
    if not isinstance(value, dict):
        raise ValueError("each review must be an object")
    return ReviewApproval(
        domain=ReviewDomain(str(value["domain"])),
        approved=bool(value["approved"]),
        reviewer=str(value["reviewer"]),
        evidence_reference=str(value["evidence_reference"]),
    )


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("acceptance plan coverage fields must be lists")
    return value
