"""Deterministic sea-trial coverage, metric, evidence, and review gates."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ReviewApproval, ReviewDomain, SeaTrialPlan, SeaTrialResult


@dataclass(frozen=True, slots=True)
class AcceptanceAudit:
    plan_id: str
    trial_count: int
    missing_coverage: tuple[str, ...]
    metric_failures: tuple[str, ...]
    missing_reviews: tuple[str, ...]
    rejected_reviews: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing_coverage or self.metric_failures or self.missing_reviews or self.rejected_reviews)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "trial_count": self.trial_count,
            "missing_coverage": list(self.missing_coverage),
            "metric_failures": list(self.metric_failures),
            "missing_reviews": list(self.missing_reviews),
            "rejected_reviews": list(self.rejected_reviews),
            "passed": self.passed,
        }


def audit_acceptance(
    plan: SeaTrialPlan,
    trials: tuple[SeaTrialResult, ...],
    reviews: tuple[ReviewApproval, ...],
) -> AcceptanceAudit:
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ValueError("sea-trial IDs must be unique")
    if len({review.domain for review in reviews}) != len(reviews):
        raise ValueError("review domains must be unique")
    missing: list[str] = []
    if len({trial.location_id for trial in trials}) < plan.minimum_locations:
        missing.append("locations")
    for time_band in plan.required_time_bands:
        if not any(trial.time_band is time_band for trial in trials):
            missing.append(f"time:{time_band.value}")
    for altitude_band in plan.required_altitude_bands:
        if not any(trial.altitude_band == altitude_band for trial in trials):
            missing.append(f"altitude:{altitude_band}")
    for weather in plan.required_weather:
        if not any(trial.weather is weather for trial in trials):
            missing.append(f"weather:{weather.value}")
    for sea_state in plan.required_sea_states:
        if not any(trial.sea_state is sea_state for trial in trials):
            missing.append(f"sea_state:{sea_state.value}")
    if sum(trial.blind_operator for trial in trials) < plan.minimum_blind_operator_trials:
        missing.append("blind_operator_trials")

    metric_failures: list[str] = []
    for trial in trials:
        if trial.false_alarms_per_hour > plan.maximum_false_alarms_per_hour:
            metric_failures.append(f"{trial.trial_id}:false_alarms_per_hour")
        if trial.missed_alerts > plan.maximum_missed_alerts:
            metric_failures.append(f"{trial.trial_id}:missed_alerts")
        if trial.reacquisition_rate < plan.minimum_reacquisition_rate:
            metric_failures.append(f"{trial.trial_id}:reacquisition_rate")
        if trial.distance_mae_m > plan.maximum_distance_mae_m:
            metric_failures.append(f"{trial.trial_id}:distance_mae_m")
    reviews_by_domain = {review.domain: review for review in reviews}
    missing_reviews = tuple(domain.value for domain in ReviewDomain if domain not in reviews_by_domain)
    rejected_reviews = tuple(
        domain.value
        for domain, review in sorted(reviews_by_domain.items(), key=lambda item: item[0].value)
        if not review.approved
    )
    return AcceptanceAudit(
        plan.plan_id,
        len(trials),
        tuple(missing),
        tuple(metric_failures),
        missing_reviews,
        rejected_reviews,
    )
