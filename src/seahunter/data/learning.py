"""Active-learning and temporally filtered teacher-student pseudo labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite

from .models import WeatherCondition


@dataclass(frozen=True, slots=True)
class ActiveLearningCandidate:
    sample_id: str
    video_id: str
    confidence_uncertainty: float
    model_disagreement: float
    fragmentation_count: int
    false_alarm: bool

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.video_id.strip():
            raise ValueError("active-learning identity fields must not be empty")
        for value in (self.confidence_uncertainty, self.model_disagreement):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("active-learning uncertainty values must be within [0, 1]")
        if self.fragmentation_count < 0:
            raise ValueError("fragmentation_count must be non-negative")


@dataclass(frozen=True, slots=True)
class ActiveLearningSelection:
    sample_id: str
    video_id: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActiveLearningConfig:
    uncertainty_weight: float = 0.3
    disagreement_weight: float = 0.3
    fragmentation_weight: float = 0.2
    false_alarm_weight: float = 0.2
    fragmentation_saturation: int = 4
    maximum_per_video: int = 20

    def __post_init__(self) -> None:
        weights = (
            self.uncertainty_weight,
            self.disagreement_weight,
            self.fragmentation_weight,
            self.false_alarm_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("active-learning weights must be non-negative with a positive sum")
        if self.fragmentation_saturation <= 0 or self.maximum_per_video <= 0:
            raise ValueError("active-learning caps must be positive")


def select_active_learning_samples(
    candidates: tuple[ActiveLearningCandidate, ...],
    budget: int,
    config: ActiveLearningConfig | None = None,
) -> tuple[ActiveLearningSelection, ...]:
    if budget < 0:
        raise ValueError("active-learning budget must be non-negative")
    settings = config or ActiveLearningConfig()
    scored: list[ActiveLearningSelection] = []
    total_weight = (
        settings.uncertainty_weight
        + settings.disagreement_weight
        + settings.fragmentation_weight
        + settings.false_alarm_weight
    )
    for candidate in candidates:
        fragmentation = min(1.0, candidate.fragmentation_count / settings.fragmentation_saturation)
        false_alarm = 1.0 if candidate.false_alarm else 0.0
        score = (
            settings.uncertainty_weight * candidate.confidence_uncertainty
            + settings.disagreement_weight * candidate.model_disagreement
            + settings.fragmentation_weight * fragmentation
            + settings.false_alarm_weight * false_alarm
        ) / total_weight
        reasons: list[str] = []
        if candidate.confidence_uncertainty >= 0.5:
            reasons.append("low_confidence")
        if candidate.model_disagreement >= 0.5:
            reasons.append("model_disagreement")
        if candidate.fragmentation_count > 0:
            reasons.append("track_fragmentation")
        if candidate.false_alarm:
            reasons.append("false_alarm")
        scored.append(ActiveLearningSelection(candidate.sample_id, candidate.video_id, score, tuple(reasons)))
    selected: list[ActiveLearningSelection] = []
    per_video: Counter[str] = Counter()
    for item in sorted(scored, key=lambda value: (-value.score, value.sample_id)):
        if len(selected) >= budget:
            break
        if per_video[item.video_id] >= settings.maximum_per_video:
            continue
        selected.append(item)
        per_video[item.video_id] += 1
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class PseudoLabel:
    sample_id: str
    video_id: str
    frame_index: int
    track_id: int
    class_name: str
    student_class_name: str
    confidence: float
    temporal_iou: float
    weather: WeatherCondition

    def __post_init__(self) -> None:
        if not all(
            value.strip() for value in (self.sample_id, self.video_id, self.class_name, self.student_class_name)
        ):
            raise ValueError("pseudo-label identity fields must not be empty")
        if self.frame_index < 0 or self.track_id < 0:
            raise ValueError("pseudo-label frame_index and track_id must be non-negative")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.temporal_iou <= 1.0:
            raise ValueError("pseudo-label confidence and temporal_iou must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class PseudoLabelFilterConfig:
    default_confidence: float = 0.7
    weather_confidence: tuple[tuple[WeatherCondition, float], ...] = ()
    minimum_temporal_iou: float = 0.3
    minimum_track_length: int = 3

    def __post_init__(self) -> None:
        thresholds = (self.default_confidence, self.minimum_temporal_iou) + tuple(
            value for _, value in self.weather_confidence
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("pseudo-label thresholds must be within [0, 1]")
        if self.minimum_track_length <= 0:
            raise ValueError("minimum_track_length must be positive")
        if len({weather for weather, _ in self.weather_confidence}) != len(self.weather_confidence):
            raise ValueError("weather confidence thresholds must be unique")


@dataclass(frozen=True, slots=True)
class PseudoLabelDecision:
    label: PseudoLabel
    accepted: bool
    rejection_reasons: tuple[str, ...]


def filter_pseudo_labels(
    labels: tuple[PseudoLabel, ...], config: PseudoLabelFilterConfig | None = None
) -> tuple[PseudoLabelDecision, ...]:
    settings = config or PseudoLabelFilterConfig()
    thresholds = dict(settings.weather_confidence)
    preliminary: dict[str, list[str]] = {}
    track_candidates: dict[tuple[str, int], list[PseudoLabel]] = defaultdict(list)
    for label in labels:
        reasons: list[str] = []
        if label.confidence < thresholds.get(label.weather, settings.default_confidence):
            reasons.append("low_confidence")
        if label.class_name != label.student_class_name:
            reasons.append("teacher_student_disagreement")
        if label.temporal_iou < settings.minimum_temporal_iou:
            reasons.append("temporal_inconsistency")
        preliminary[label.sample_id] = reasons
        if not reasons:
            track_candidates[(label.video_id, label.track_id)].append(label)

    for track_labels in track_candidates.values():
        if len(track_labels) < settings.minimum_track_length:
            for label in track_labels:
                preliminary[label.sample_id].append("short_tracklet")
            continue
        classes = Counter(label.class_name for label in track_labels)
        if len(classes) > 1:
            for label in track_labels:
                preliminary[label.sample_id].append("tracklet_class_inconsistency")
    return tuple(
        PseudoLabelDecision(label, not preliminary[label.sample_id], tuple(preliminary[label.sample_id]))
        for label in labels
    )
