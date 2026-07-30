"""Calibrated detection and tracking confidence fusion with bounded memory."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log


@dataclass(frozen=True, slots=True)
class JointConfidenceEvidence:
    detection_probability: float
    track_prior: float
    motion_consistency: float
    reid_similarity: float | None
    state_uncertainty: float
    wave_periodicity: float
    consecutive_observations: int

    def __post_init__(self) -> None:
        values = (
            self.detection_probability,
            self.track_prior,
            self.motion_consistency,
            self.state_uncertainty,
            self.wave_periodicity,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("joint-confidence evidence must be finite and within [0, 1]")
        if self.reid_similarity is not None and not 0.0 <= self.reid_similarity <= 1.0:
            raise ValueError("ReID similarity must be within [0, 1]")
        if self.consecutive_observations <= 0:
            raise ValueError("consecutive_observations must be positive")


@dataclass(frozen=True, slots=True)
class JointConfidenceConfig:
    temperature: float = 1.0
    detection_weight: float = 0.35
    track_weight: float = 0.2
    motion_weight: float = 0.2
    reid_weight: float = 0.1
    uncertainty_weight: float = 0.15
    persistence_saturation: int = 5
    maximum_persistence_boost: float = 0.1
    wave_veto_threshold: float = 0.8
    wave_confidence_cap: float = 0.25

    def __post_init__(self) -> None:
        if not isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("confidence temperature must be finite and positive")
        weights = (
            self.detection_weight,
            self.track_weight,
            self.motion_weight,
            self.reid_weight,
            self.uncertainty_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("joint-confidence weights must be non-negative with positive sum")
        if self.persistence_saturation <= 0 or not 0.0 <= self.maximum_persistence_boost <= 1.0:
            raise ValueError("confidence persistence controls are invalid")
        if not 0.0 <= self.wave_veto_threshold <= 1.0 or not 0.0 <= self.wave_confidence_cap <= 1.0:
            raise ValueError("wave-veto controls must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class JointConfidenceResult:
    confidence: float
    calibrated_detection: float
    persistence_boost: float
    vetoed_by_wave: bool
    components: tuple[tuple[str, float], ...]


class JointConfidenceUpdater:
    def __init__(self, config: JointConfidenceConfig | None = None) -> None:
        self.config = config or JointConfidenceConfig()

    def update(self, evidence: JointConfidenceEvidence) -> JointConfidenceResult:
        detection = _temperature_scale_probability(evidence.detection_probability, self.config.temperature)
        reid = evidence.reid_similarity
        components = [
            ("detection", detection, self.config.detection_weight),
            ("track", evidence.track_prior, self.config.track_weight),
            ("motion", evidence.motion_consistency, self.config.motion_weight),
            ("certainty", 1.0 - evidence.state_uncertainty, self.config.uncertainty_weight),
        ]
        if reid is not None:
            components.append(("reid", reid, self.config.reid_weight))
        numerator = sum(value * weight for _, value, weight in components)
        denominator = sum(weight for _, _, weight in components)
        base = numerator / denominator
        persistence_fraction = min(1.0, evidence.consecutive_observations / self.config.persistence_saturation)
        persistence_boost = self.config.maximum_persistence_boost * persistence_fraction
        confidence = min(1.0, base + persistence_boost)
        vetoed = evidence.wave_periodicity >= self.config.wave_veto_threshold
        if vetoed:
            confidence = min(confidence, self.config.wave_confidence_cap)
        return JointConfidenceResult(
            confidence=confidence,
            calibrated_detection=detection,
            persistence_boost=persistence_boost,
            vetoed_by_wave=vetoed,
            components=tuple((name, value) for name, value, _ in components),
        )


def _temperature_scale_probability(probability: float, temperature: float) -> float:
    epsilon = 1e-9
    bounded = min(1.0 - epsilon, max(epsilon, probability))
    logit = log(bounded / (1.0 - bounded)) / temperature
    if logit >= 0.0:
        return 1.0 / (1.0 + exp(-logit))
    positive = exp(logit)
    return positive / (1.0 + positive)
