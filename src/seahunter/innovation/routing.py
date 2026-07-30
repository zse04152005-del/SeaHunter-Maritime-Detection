"""Clip-stable degradation-aware preprocessing and threshold routing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from seahunter.data import WeatherCondition


@dataclass(frozen=True, slots=True)
class RouteProfile:
    weather: WeatherCondition
    model_id: str
    preprocessor: str
    detection_threshold: float

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.preprocessor.strip():
            raise ValueError("route model_id and preprocessor must not be empty")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("route detection_threshold must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class DegradationObservation:
    clip_id: str
    frame_index: int
    weather: WeatherCondition
    intensity: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.clip_id.strip() or self.frame_index < 0:
            raise ValueError("degradation observation identity is invalid")
        if not 0 <= self.intensity <= 3 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("degradation intensity or confidence is invalid")


@dataclass(frozen=True, slots=True)
class WeatherRouterConfig:
    profiles: tuple[RouteProfile, ...]
    voting_window: int = 12
    minimum_votes: int = 8
    minimum_dwell_frames: int = 30
    minimum_classifier_confidence: float = 0.6
    allow_expert_models: bool = False

    def __post_init__(self) -> None:
        if len({profile.weather for profile in self.profiles}) != len(self.profiles):
            raise ValueError("route profiles must have unique weather conditions")
        if {profile.weather for profile in self.profiles} != set(WeatherCondition):
            raise ValueError("route profiles must cover every weather condition")
        if not 1 <= self.minimum_votes <= self.voting_window or self.minimum_dwell_frames < 0:
            raise ValueError("weather routing voting and dwell controls are invalid")
        if not 0.0 <= self.minimum_classifier_confidence <= 1.0:
            raise ValueError("minimum classifier confidence must be within [0, 1]")
        model_ids = {profile.model_id for profile in self.profiles}
        if not self.allow_expert_models and len(model_ids) != 1:
            raise ValueError("single robust model is required unless expert routing is explicitly enabled")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    profile: RouteProfile
    switched: bool
    reason: str


class WeatherRouter:
    def __init__(self, config: WeatherRouterConfig) -> None:
        self.config = config
        self._profiles = {profile.weather: profile for profile in config.profiles}
        self._clip_id: str | None = None
        self._votes: deque[WeatherCondition] = deque(maxlen=config.voting_window)
        self._active = WeatherCondition.NORMAL
        self._last_switch_frame = 0
        self._last_frame = -1

    def update(self, observation: DegradationObservation) -> RoutingDecision:
        if self._clip_id != observation.clip_id:
            self._clip_id = observation.clip_id
            self._votes.clear()
            self._active = WeatherCondition.NORMAL
            self._last_switch_frame = observation.frame_index
            self._last_frame = -1
        if observation.frame_index <= self._last_frame:
            raise ValueError("weather observations must increase strictly within a clip")
        self._last_frame = observation.frame_index
        if observation.confidence >= self.config.minimum_classifier_confidence:
            self._votes.append(observation.weather)
        counts = Counter(self._votes)
        candidate, votes = counts.most_common(1)[0] if counts else (self._active, 0)
        dwell = observation.frame_index - self._last_switch_frame
        if (
            candidate is not self._active
            and votes >= self.config.minimum_votes
            and dwell >= self.config.minimum_dwell_frames
        ):
            self._active = candidate
            self._last_switch_frame = observation.frame_index
            return RoutingDecision(self._profiles[self._active], True, "stable_clip_vote")
        reason = (
            "insufficient_classifier_confidence"
            if observation.confidence < self.config.minimum_classifier_confidence
            else "held_by_hysteresis"
        )
        return RoutingDecision(self._profiles[self._active], False, reason)


def needs_sim2real(real_samples_by_weather: dict[WeatherCondition, int], *, minimum_real_samples: int) -> bool:
    if minimum_real_samples <= 0:
        raise ValueError("minimum_real_samples must be positive")
    return any(real_samples_by_weather.get(weather, 0) < minimum_real_samples for weather in WeatherCondition)
