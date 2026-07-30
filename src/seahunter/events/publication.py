"""Bounded WebSocket event hub and MQTT JSON publisher."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Any, Protocol

from seahunter.schemas import RiskEvent

from .persistence import risk_event_to_dict


@dataclass(frozen=True, slots=True)
class PublishedEvent:
    sequence: int
    event: RiskEvent


class EventHub:
    def __init__(self, maximum_history: int = 1024) -> None:
        if maximum_history <= 0:
            raise ValueError("maximum_history must be positive")
        self._condition = Condition()
        self._history: deque[PublishedEvent] = deque(maxlen=maximum_history)
        self._next_sequence = 0
        self._closed = False

    def publish(self, event: RiskEvent) -> PublishedEvent:
        with self._condition:
            if self._closed:
                raise RuntimeError("event hub is closed")
            published = PublishedEvent(self._next_sequence, event)
            self._next_sequence += 1
            self._history.append(published)
            self._condition.notify_all()
            return published

    def wait_for_next(self, after_sequence: int, timeout: float | None = None) -> PublishedEvent | None:
        if after_sequence < -1:
            raise ValueError("after_sequence must be at least -1")
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while not any(item.sequence > after_sequence for item in self._history) and not self._closed:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            return next((item for item in self._history if item.sequence > after_sequence), None)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


class MQTTClient(Protocol):
    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> Any: ...


class MQTTEventPublisher:
    def __init__(self, client: MQTTClient, *, topic_prefix: str = "seahunter/events", qos: int = 1) -> None:
        if not topic_prefix.strip("/"):
            raise ValueError("topic_prefix must not be empty")
        if qos not in (0, 1, 2):
            raise ValueError("MQTT qos must be 0, 1, or 2")
        self.client = client
        self.topic_prefix = topic_prefix.strip("/")
        self.qos = qos

    def publish(self, event: RiskEvent) -> None:
        topic = f"{self.topic_prefix}/{event.zone_id}/{event.rule_id}"
        payload = json.dumps(risk_event_to_dict(event), sort_keys=True, separators=(",", ":"))
        self.client.publish(topic, payload, qos=self.qos, retain=False)
