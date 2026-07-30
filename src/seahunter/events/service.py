"""Durable event orchestration for edge and API consumers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from seahunter.schemas import EventStatus, RiskEvent

from .engine import DangerZoneEngine, ZoneObservation
from .evidence import EvidenceFrame, EvidenceRecorder
from .feedback import HardSampleFeedbackWriter
from .persistence import FeedbackLabel, OperatorFeedback, SQLiteEventStore
from .publication import EventHub, MQTTEventPublisher


@dataclass(frozen=True, slots=True)
class PublicationFailure:
    event_id: str
    publisher: str
    failed_at: datetime
    error: str


class EventService:
    """Coordinate local-first event state, evidence, APIs, and optional MQTT."""

    def __init__(
        self,
        engine: DangerZoneEngine,
        store: SQLiteEventStore,
        evidence: EvidenceRecorder,
        hub: EventHub | None = None,
        *,
        mqtt_publishers: tuple[MQTTEventPublisher, ...] = (),
        hard_samples: HardSampleFeedbackWriter | None = None,
        maximum_publication_failures: int = 256,
    ) -> None:
        if maximum_publication_failures <= 0:
            raise ValueError("maximum_publication_failures must be positive")
        self.engine = engine
        self.store = store
        self.evidence = evidence
        self.hub = hub or EventHub()
        self.mqtt_publishers = mqtt_publishers
        self.hard_samples = hard_samples
        self._publication_failures: deque[PublicationFailure] = deque(maxlen=maximum_publication_failures)
        self._lock = RLock()
        self._closed = False

        active_events = self.store.list_events(active_only=True)
        self.engine.restore_active_events(active_events)
        for event in active_events:
            if not self.evidence.bundle_path(event.event_id).exists():
                self.evidence.open_event(event)

    def update(self, observation: ZoneObservation) -> tuple[RiskEvent, ...]:
        with self._lock:
            self._ensure_open()
            emitted: list[RiskEvent] = []
            for event in self.engine.update(observation):
                prepared = self._attach_evidence_to_open_event(event)
                self._record_and_publish(prepared)
                emitted.append(prepared)
            return tuple(emitted)

    def ingest_evidence(self, frame: EvidenceFrame) -> tuple[Path, ...]:
        with self._lock:
            self._ensure_open()
            return self.evidence.ingest(frame)

    def acknowledge(self, event_id: str, acknowledged_at: datetime | None = None) -> RiskEvent:
        with self._lock:
            self._ensure_open()
            event = self.engine.acknowledge(event_id, acknowledged_at or datetime.now(timezone.utc))
            self._record_and_publish(event)
            return event

    def add_feedback(self, feedback: OperatorFeedback) -> OperatorFeedback:
        with self._lock:
            self._ensure_open()
            event = self.store.get(feedback.event_id)
            if event is None:
                raise KeyError("feedback event_id not found")
            self.store.add_feedback(feedback)
            if feedback.label is FeedbackLabel.FALSE_POSITIVE and self.hard_samples is not None:
                self.hard_samples.write(event, feedback)
            return feedback

    def get_event(self, event_id: str) -> RiskEvent | None:
        with self._lock:
            self._ensure_open()
            return self.store.get(event_id)

    def list_events(self, *, active_only: bool = False) -> tuple[RiskEvent, ...]:
        with self._lock:
            self._ensure_open()
            return self.store.list_events(active_only=active_only)

    def list_transitions(self, event_id: str) -> tuple[RiskEvent, ...]:
        with self._lock:
            self._ensure_open()
            return self.store.list_transitions(event_id)

    def list_feedback(self, event_id: str | None = None) -> tuple[OperatorFeedback, ...]:
        with self._lock:
            self._ensure_open()
            return self.store.list_feedback(event_id)

    def publication_failures(self) -> tuple[PublicationFailure, ...]:
        with self._lock:
            return tuple(self._publication_failures)

    def close(self) -> tuple[Path, ...]:
        with self._lock:
            if self._closed:
                return ()
            completed = self.evidence.close()
            self.hub.close()
            self.store.close()
            self._closed = True
            return completed

    def _attach_evidence_to_open_event(self, event: RiskEvent) -> RiskEvent:
        if event.status is not EventStatus.OPEN or event.evidence_uri is not None:
            return event
        path = self.evidence.bundle_path(event.event_id).resolve()
        updated = self.engine.attach_evidence(event.event_id, path.as_uri())
        self.evidence.open_event(updated)
        return updated

    def _record_and_publish(self, event: RiskEvent) -> None:
        self.evidence.update_event(event)
        self.store.append(event)
        self.hub.publish(event)
        for publisher in self.mqtt_publishers:
            try:
                publisher.publish(event)
            except Exception as exc:  # MQTT failure must not stop local warning or evidence capture.
                self._publication_failures.append(
                    PublicationFailure(
                        event_id=event.event_id,
                        publisher=type(publisher).__name__,
                        failed_at=datetime.now(timezone.utc),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("event service is closed")
