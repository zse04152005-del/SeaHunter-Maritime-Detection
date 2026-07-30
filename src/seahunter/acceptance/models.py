"""Sea-trial, operator, compliance, and release acceptance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from seahunter.data import WeatherCondition


class TimeBand(str, Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


class SeaState(str, Enum):
    CALM = "calm"
    MODERATE = "moderate"
    ROUGH = "rough"


class ReviewDomain(str, Enum):
    SAFETY = "safety"
    PRIVACY = "privacy"
    LICENSE = "license"
    OPERATIONS = "operations"


@dataclass(frozen=True, slots=True)
class SeaTrialPlan:
    plan_id: str
    minimum_locations: int
    required_time_bands: tuple[TimeBand, ...]
    required_altitude_bands: tuple[str, ...]
    required_weather: tuple[WeatherCondition, ...]
    required_sea_states: tuple[SeaState, ...]
    minimum_blind_operator_trials: int
    maximum_false_alarms_per_hour: float
    maximum_missed_alerts: int
    minimum_reacquisition_rate: float
    maximum_distance_mae_m: float

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or self.minimum_locations <= 0 or self.minimum_blind_operator_trials <= 0:
            raise ValueError("sea-trial plan identity and minimum counts are invalid")
        if not self.required_time_bands or not self.required_altitude_bands or not self.required_weather:
            raise ValueError("sea-trial coverage requirements must not be empty")
        if not self.required_sea_states or any(not band.strip() for band in self.required_altitude_bands):
            raise ValueError("sea-state and altitude requirements must not be empty")
        if len(set(self.required_altitude_bands)) != len(self.required_altitude_bands):
            raise ValueError("required altitude bands must be unique")
        if self.maximum_false_alarms_per_hour < 0.0 or self.maximum_missed_alerts < 0:
            raise ValueError("sea-trial maximum error thresholds must be non-negative")
        if not 0.0 <= self.minimum_reacquisition_rate <= 1.0 or self.maximum_distance_mae_m < 0.0:
            raise ValueError("sea-trial reacquisition and distance thresholds are invalid")


@dataclass(frozen=True, slots=True)
class SeaTrialResult:
    trial_id: str
    location_id: str
    time_band: TimeBand
    altitude_band: str
    weather: WeatherCondition
    sea_state: SeaState
    blind_operator: bool
    operator_id: str
    calibration_report: str
    telemetry_sync_report: str
    zone_config_version: str
    event_audit_reference: str
    evidence_bundle_reference: str
    model_package_sha256: str
    false_alarms_per_hour: float
    missed_alerts: int
    reacquisition_rate: float
    distance_mae_m: float

    def __post_init__(self) -> None:
        identity = (
            self.trial_id,
            self.location_id,
            self.altitude_band,
            self.operator_id,
            self.calibration_report,
            self.telemetry_sync_report,
            self.zone_config_version,
            self.event_audit_reference,
            self.evidence_bundle_reference,
        )
        if not all(value.strip() for value in identity):
            raise ValueError("sea-trial identity and evidence fields must not be empty")
        if len(self.model_package_sha256) != 64:
            raise ValueError("sea-trial model package reference must be a SHA-256 digest")
        numeric = (self.false_alarms_per_hour, self.reacquisition_rate, self.distance_mae_m)
        if any(not isfinite(value) or value < 0.0 for value in numeric) or self.missed_alerts < 0:
            raise ValueError("sea-trial metrics must be finite and non-negative")
        if self.reacquisition_rate > 1.0:
            raise ValueError("sea-trial reacquisition_rate must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ReviewApproval:
    domain: ReviewDomain
    approved: bool
    reviewer: str
    evidence_reference: str

    def __post_init__(self) -> None:
        if not self.reviewer.strip() or not self.evidence_reference.strip():
            raise ValueError("reviewer and review evidence must not be empty")


@dataclass(frozen=True, slots=True)
class ExternalReleaseGates:
    geolocation_truth_set_passed: bool
    real_maritime_metrics_passed: bool
    tensorrt_target_passed: bool
    nvdec_target_passed: bool
    soak_72h_passed: bool
    live_fault_matrix_passed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.geolocation_truth_set_passed,
                self.real_maritime_metrics_passed,
                self.tensorrt_target_passed,
                self.nvdec_target_passed,
                self.soak_72h_passed,
                self.live_fault_matrix_passed,
            )
        )
