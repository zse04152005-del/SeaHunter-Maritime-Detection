"""Dependency-light confidence calibration and weather-threshold evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

from .models import WeatherCondition


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    logit: float
    positive: bool
    weather: WeatherCondition

    def __post_init__(self) -> None:
        if not isfinite(self.logit):
            raise ValueError("calibration logit must be finite")


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    temperature: float
    negative_log_likelihood: float
    expected_calibration_error: float

    def probability(self, logit: float) -> float:
        return _sigmoid(logit / self.temperature)


def fit_temperature(
    samples: tuple[CalibrationSample, ...],
    *,
    candidates: tuple[float, ...] | None = None,
) -> TemperatureCalibration:
    if not samples:
        raise ValueError("temperature calibration requires samples")
    if candidates is None:
        candidates = tuple(index / 10.0 for index in range(5, 51))
    if not candidates or any(not isfinite(value) or value <= 0.0 for value in candidates):
        raise ValueError("temperature candidates must be finite and positive")
    losses = [(temperature, _negative_log_likelihood(samples, temperature)) for temperature in candidates]
    temperature, loss = min(losses, key=lambda item: (item[1], item[0]))
    probabilities = tuple(_sigmoid(sample.logit / temperature) for sample in samples)
    return TemperatureCalibration(temperature, loss, expected_calibration_error(samples, probabilities))


def expected_calibration_error(
    samples: tuple[CalibrationSample, ...], probabilities: tuple[float, ...], *, bins: int = 10
) -> float:
    if len(samples) != len(probabilities) or not samples:
        raise ValueError("ECE samples and probabilities must have the same non-zero length")
    if bins <= 0 or any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("ECE bins and probabilities are invalid")
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[item] for item in members) / len(members)
        accuracy = sum(1.0 if samples[item].positive else 0.0 for item in members) / len(members)
        error += len(members) / len(samples) * abs(confidence - accuracy)
    return error


@dataclass(frozen=True, slots=True)
class ThresholdSample:
    probability: float
    positive: bool
    weather: WeatherCondition

    def __post_init__(self) -> None:
        if not isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("threshold probability must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class WeatherThresholdResult:
    weather: WeatherCondition
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    support: int


def evaluate_weather_thresholds(
    samples: tuple[ThresholdSample, ...],
    *,
    minimum_precision: float = 0.8,
    candidates: tuple[float, ...] | None = None,
) -> tuple[WeatherThresholdResult, ...]:
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be within [0, 1]")
    if candidates is None:
        candidates = tuple(index / 20.0 for index in range(1, 20))
    if not candidates or any(not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("threshold candidates must be within [0, 1]")
    results: list[WeatherThresholdResult] = []
    for weather in WeatherCondition:
        subset = tuple(sample for sample in samples if sample.weather is weather)
        if not subset:
            continue
        evaluated = tuple(_threshold_metrics(subset, weather, threshold) for threshold in candidates)
        eligible = tuple(item for item in evaluated if item.precision >= minimum_precision)
        pool = eligible or evaluated
        results.append(max(pool, key=lambda item: (item.f1, item.recall, item.precision, item.threshold)))
    return tuple(results)


def _threshold_metrics(
    samples: tuple[ThresholdSample, ...], weather: WeatherCondition, threshold: float
) -> WeatherThresholdResult:
    true_positives = sum(sample.positive and sample.probability >= threshold for sample in samples)
    false_positives = sum(not sample.positive and sample.probability >= threshold for sample in samples)
    false_negatives = sum(sample.positive and sample.probability < threshold for sample in samples)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return WeatherThresholdResult(weather, threshold, precision, recall, f1, false_positives, len(samples))


def _negative_log_likelihood(samples: tuple[CalibrationSample, ...], temperature: float) -> float:
    epsilon = 1e-12
    total = 0.0
    for sample in samples:
        probability = min(1.0 - epsilon, max(epsilon, _sigmoid(sample.logit / temperature)))
        total -= log(probability if sample.positive else 1.0 - probability)
    return total / len(samples)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    positive = exp(value)
    return positive / (1.0 + positive)
