"""Stateful danger-zone rules, risk scoring, and event lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from math import hypot
from uuid import NAMESPACE_URL, uuid5

from seahunter.schemas import EventStatus, MotionTrend, RiskEvent

from .zones import DangerZone, signed_distance_to_polygon


class ZoneRule(str, Enum):
    ENTER = "enter"
    EXIT = "exit"
    DWELL = "dwell"
    APPROACH = "approach"
    REVERSE = "reverse"
    COLLISION = "collision"


@dataclass(frozen=True, slots=True)
class ZoneObservation:
    track_id: int
    captured_at: datetime
    point: tuple[float, float]
    velocity_en_m_s: tuple[float, float]
    reliability: float
    class_id: int
    ttc_seconds: float | None = None
    trend: MotionTrend = MotionTrend.UNKNOWN

    def __post_init__(self) -> None:
        if self.track_id < 0 or self.class_id < 0:
            raise ValueError("track_id and class_id must be non-negative")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0, 1]")
        if self.ttc_seconds is not None and self.ttc_seconds < 0.0:
            raise ValueError("ttc_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class DangerZoneEngineConfig:
    enter_margin_m: float = 2.0
    exit_margin_m: float = 3.0
    minimum_enter_seconds: float = 1.0
    minimum_exit_seconds: float = 1.0
    minimum_dwell_seconds: float = 10.0
    cooldown_seconds: float = 30.0
    minimum_reliability: float = 0.5
    minimum_direction_speed_m_s: float = 0.5
    reverse_cosine_threshold: float = 0.5
    approach_progress_m: float = 0.5
    ttc_risk_horizon_seconds: float = 60.0
    reliability_weight: float = 0.4
    ttc_weight: float = 0.25
    class_weight: float = 0.15
    zone_weight: float = 0.2
    class_risk: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        non_negative = (
            self.enter_margin_m,
            self.exit_margin_m,
            self.minimum_enter_seconds,
            self.minimum_exit_seconds,
            self.minimum_dwell_seconds,
            self.cooldown_seconds,
            self.minimum_direction_speed_m_s,
            self.approach_progress_m,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("event timing, margin, and speed controls must be non-negative")
        if not 0.0 <= self.minimum_reliability <= 1.0:
            raise ValueError("minimum_reliability must be within [0, 1]")
        if not 0.0 <= self.reverse_cosine_threshold <= 1.0:
            raise ValueError("reverse_cosine_threshold must be within [0, 1]")
        if self.ttc_risk_horizon_seconds <= 0.0:
            raise ValueError("ttc_risk_horizon_seconds must be positive")
        weights = (self.reliability_weight, self.ttc_weight, self.class_weight, self.zone_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("risk weights must be non-negative with a positive sum")
        if len({class_id for class_id, _ in self.class_risk}) != len(self.class_risk):
            raise ValueError("class_risk IDs must be unique")
        if any(class_id < 0 or not 0.0 <= risk <= 1.0 for class_id, risk in self.class_risk):
            raise ValueError("class_risk entries must contain non-negative IDs and risks within [0, 1]")


@dataclass(slots=True)
class _ZoneTrackState:
    stable_inside: bool = False
    inside_candidate_since: datetime | None = None
    outside_candidate_since: datetime | None = None
    entered_at: datetime | None = None
    last_outside_distance_m: float | None = None
    last_captured_at: datetime | None = None


class DangerZoneEngine:
    def __init__(self, zones: tuple[DangerZone, ...], config: DangerZoneEngineConfig | None = None) -> None:
        if not zones:
            raise ValueError("at least one danger zone is required")
        if len({zone.zone_id for zone in zones}) != len(zones):
            raise ValueError("danger zone IDs must be unique")
        self.zones = zones
        self.config = config or DangerZoneEngineConfig()
        self._states: dict[tuple[int, str], _ZoneTrackState] = {}
        self._active: dict[tuple[int, str, ZoneRule], RiskEvent] = {}
        self._events_by_id: dict[str, RiskEvent] = {}
        self._cooldown_until: dict[tuple[int, str, ZoneRule], datetime] = {}

    def update(self, observation: ZoneObservation) -> list[RiskEvent]:
        if observation.reliability < self.config.minimum_reliability:
            return []
        emitted: list[RiskEvent] = []
        for zone in self.zones:
            key = (observation.track_id, zone.zone_id)
            state = self._states.setdefault(key, _ZoneTrackState())
            if state.last_captured_at is not None and observation.captured_at <= state.last_captured_at:
                raise ValueError("zone observations must increase strictly per track and zone")
            point_local = zone.point_to_local(observation.point)
            polygon_local = zone.local_polygon()
            signed_distance = signed_distance_to_polygon(point_local, polygon_local)
            was_inside = state.stable_inside
            self._update_hysteresis(state, signed_distance, observation.captured_at)
            entered_now = state.stable_inside and not was_inside
            exited_now = was_inside and not state.stable_inside
            if entered_now:
                state.entered_at = observation.captured_at
            if exited_now:
                state.entered_at = None

            outside_distance = max(0.0, -signed_distance)
            approaching = (
                not state.stable_inside
                and outside_distance <= zone.approach_distance_m
                and state.last_outside_distance_m is not None
                and state.last_outside_distance_m - outside_distance >= self.config.approach_progress_m
            )
            reverse = self._is_reverse(zone, observation) and state.stable_inside
            collision = (
                observation.ttc_seconds is not None
                and observation.ttc_seconds <= zone.collision_ttc_seconds
                and observation.trend is MotionTrend.APPROACHING
            )
            dwelling = (
                state.stable_inside
                and state.entered_at is not None
                and (observation.captured_at - state.entered_at).total_seconds() >= self.config.minimum_dwell_seconds
            )
            conditions = {
                ZoneRule.ENTER: state.stable_inside,
                ZoneRule.EXIT: exited_now,
                ZoneRule.DWELL: dwelling,
                ZoneRule.APPROACH: approaching,
                ZoneRule.REVERSE: reverse,
                ZoneRule.COLLISION: collision,
            }
            risk = self._risk_score(zone, observation)
            for rule, active in conditions.items():
                event = self._transition(observation, zone, rule, active, risk)
                if event is not None:
                    emitted.append(event)
            state.last_outside_distance_m = None if state.stable_inside else outside_distance
            state.last_captured_at = observation.captured_at
        return emitted

    def acknowledge(self, event_id: str, acknowledged_at: datetime) -> RiskEvent:
        event = self._events_by_id.get(event_id)
        if event is None or event.status is EventStatus.CLOSED:
            raise KeyError("active event_id not found")
        if acknowledged_at.tzinfo is None or acknowledged_at.utcoffset() is None:
            raise ValueError("acknowledged_at must be timezone-aware")
        if acknowledged_at < event.updated_at:
            raise ValueError("acknowledged_at cannot precede the event update")
        acknowledged = replace(event, status=EventStatus.ACKNOWLEDGED, updated_at=acknowledged_at)
        self._events_by_id[event_id] = acknowledged
        for key, active in self._active.items():
            if active.event_id == event_id:
                self._active[key] = acknowledged
                break
        return acknowledged

    def active_events(self) -> tuple[RiskEvent, ...]:
        return tuple(sorted(self._active.values(), key=lambda event: event.event_id))

    def _update_hysteresis(self, state: _ZoneTrackState, signed_distance: float, captured_at: datetime) -> None:
        if not state.stable_inside:
            if signed_distance >= self.config.enter_margin_m:
                state.inside_candidate_since = state.inside_candidate_since or captured_at
                if (captured_at - state.inside_candidate_since).total_seconds() >= self.config.minimum_enter_seconds:
                    state.stable_inside = True
                    state.outside_candidate_since = None
            else:
                state.inside_candidate_since = None
        else:
            if signed_distance <= -self.config.exit_margin_m:
                state.outside_candidate_since = state.outside_candidate_since or captured_at
                if (captured_at - state.outside_candidate_since).total_seconds() >= self.config.minimum_exit_seconds:
                    state.stable_inside = False
                    state.inside_candidate_since = None
            else:
                state.outside_candidate_since = None

    def _is_reverse(self, zone: DangerZone, observation: ZoneObservation) -> bool:
        if zone.expected_direction_en is None:
            return False
        speed = hypot(*observation.velocity_en_m_s)
        expected_speed = hypot(*zone.expected_direction_en)
        if speed < self.config.minimum_direction_speed_m_s:
            return False
        cosine = (
            observation.velocity_en_m_s[0] * zone.expected_direction_en[0]
            + observation.velocity_en_m_s[1] * zone.expected_direction_en[1]
        ) / (speed * expected_speed)
        return cosine <= -self.config.reverse_cosine_threshold

    def _risk_score(self, zone: DangerZone, observation: ZoneObservation) -> float:
        class_risk = dict(self.config.class_risk).get(observation.class_id, 0.5)
        ttc_risk = (
            0.0
            if observation.ttc_seconds is None
            else max(0.0, 1.0 - observation.ttc_seconds / self.config.ttc_risk_horizon_seconds)
        )
        numerator = (
            self.config.reliability_weight * observation.reliability
            + self.config.ttc_weight * ttc_risk
            + self.config.class_weight * class_risk
            + self.config.zone_weight * zone.severity
        )
        denominator = (
            self.config.reliability_weight + self.config.ttc_weight + self.config.class_weight + self.config.zone_weight
        )
        return max(0.0, min(1.0, numerator / denominator))

    def _transition(
        self,
        observation: ZoneObservation,
        zone: DangerZone,
        rule: ZoneRule,
        condition: bool,
        risk: float,
    ) -> RiskEvent | None:
        key = (observation.track_id, zone.zone_id, rule)
        current = self._active.get(key)
        if not condition:
            if current is None:
                return None
            closed = replace(current, status=EventStatus.CLOSED, updated_at=observation.captured_at)
            self._events_by_id[closed.event_id] = closed
            self._active.pop(key, None)
            self._cooldown_until[key] = observation.captured_at + timedelta(seconds=self.config.cooldown_seconds)
            return closed
        if current is None:
            cooldown_until = self._cooldown_until.get(key)
            if cooldown_until is not None and observation.captured_at < cooldown_until:
                return None
            identity = (
                f"seahunter:{zone.zone_id}:{rule.value}:{observation.track_id}:{observation.captured_at.isoformat()}"
            )
            opened = RiskEvent(
                event_id=str(uuid5(NAMESPACE_URL, identity)),
                rule_id=rule.value,
                zone_id=zone.zone_id,
                track_id=observation.track_id,
                opened_at=observation.captured_at,
                updated_at=observation.captured_at,
                status=EventStatus.OPEN,
                risk_score=risk,
            )
            self._active[key] = opened
            self._events_by_id[opened.event_id] = opened
            return opened
        status = EventStatus.ACKNOWLEDGED if current.status is EventStatus.ACKNOWLEDGED else EventStatus.UPDATED
        updated = replace(current, status=status, updated_at=observation.captured_at, risk_score=risk)
        self._active[key] = updated
        self._events_by_id[updated.event_id] = updated
        return updated
