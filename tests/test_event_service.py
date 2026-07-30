from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from seahunter.events import (
    DangerZone,
    DangerZoneEngine,
    DangerZoneEngineConfig,
    EventService,
    EvidenceRecorder,
    FeedbackLabel,
    HardSampleFeedbackWriter,
    MQTTEventPublisher,
    OperatorFeedback,
    SQLiteEventStore,
    ZoneCoordinateSpace,
    ZoneObservation,
)
from seahunter.schemas import EventStatus

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


class FakeMQTTClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[tuple[str, str, int, bool]] = []

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> Any:
        if self.fail:
            raise ConnectionError("broker unavailable")
        self.messages.append((topic, payload, qos, retain))
        return None


def make_engine() -> DangerZoneEngine:
    zone = DangerZone(
        zone_id="restricted",
        name="Restricted",
        polygon=((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
        coordinate_space=ZoneCoordinateSpace.LOCAL_ENU,
        severity=0.9,
    )
    return DangerZoneEngine(
        (zone,),
        DangerZoneEngineConfig(minimum_enter_seconds=0.0, minimum_exit_seconds=0.0),
    )


def make_observation(seconds: float, point: tuple[float, float] = (10.0, 10.0)) -> ZoneObservation:
    return ZoneObservation(
        track_id=8,
        captured_at=START + timedelta(seconds=seconds),
        point=point,
        velocity_en_m_s=(0.0, 0.0),
        reliability=0.95,
        class_id=1,
    )


class EventServiceTests(unittest.TestCase):
    def test_persists_publishes_and_keeps_mqtt_failure_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            successful_client = FakeMQTTClient()
            failed_client = FakeMQTTClient(fail=True)
            service = EventService(
                make_engine(),
                SQLiteEventStore(root / "events.sqlite3"),
                EvidenceRecorder(root / "evidence"),
                mqtt_publishers=(
                    MQTTEventPublisher(successful_client),
                    MQTTEventPublisher(failed_client),
                ),
            )

            event = next(item for item in service.update(make_observation(0.0)) if item.rule_id == "enter")

            self.assertIsNotNone(event.evidence_uri)
            self.assertEqual(service.get_event(event.event_id), event)
            self.assertEqual(service.hub.wait_for_next(-1, 0.0).event, event)  # type: ignore[union-attr]
            self.assertEqual(len(successful_client.messages), 1)
            self.assertEqual(json.loads(successful_client.messages[0][1])["event_id"], event.event_id)
            failures = service.publication_failures()
            self.assertEqual(len(failures), 1)
            self.assertIn("broker unavailable", failures[0].error)
            service.close()

    def test_acknowledgement_and_active_event_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            service = EventService(make_engine(), SQLiteEventStore(database), EvidenceRecorder(root / "evidence"))
            opened = next(item for item in service.update(make_observation(0.0)) if item.rule_id == "enter")
            acknowledged = service.acknowledge(opened.event_id, START + timedelta(seconds=1.0))
            self.assertEqual(acknowledged.status, EventStatus.ACKNOWLEDGED)
            self.assertEqual(
                [item.status for item in service.list_transitions(opened.event_id)],
                [EventStatus.OPEN, EventStatus.ACKNOWLEDGED],
            )
            service.close()

            restarted = EventService(make_engine(), SQLiteEventStore(database), EvidenceRecorder(root / "evidence"))
            self.assertEqual(restarted.engine.active_events(), (acknowledged,))
            updated = next(
                item for item in restarted.update(make_observation(2.0)) if item.event_id == acknowledged.event_id
            )
            self.assertEqual(updated.status, EventStatus.ACKNOWLEDGED)
            restarted.close()

    def test_false_positive_feedback_enters_hard_sample_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hard_samples = root / "hard-samples.jsonl"
            service = EventService(
                make_engine(),
                SQLiteEventStore(root / "events.sqlite3"),
                EvidenceRecorder(root / "evidence"),
                hard_samples=HardSampleFeedbackWriter(hard_samples),
            )
            event = next(item for item in service.update(make_observation(0.0)) if item.rule_id == "enter")
            feedback = OperatorFeedback(
                event_id=event.event_id,
                label=FeedbackLabel.FALSE_POSITIVE,
                operator_id="operator-7",
                created_at=START + timedelta(seconds=3.0),
                note="wave glint",
            )

            service.add_feedback(feedback)

            self.assertEqual(service.list_feedback(event.event_id), (feedback,))
            record = json.loads(hard_samples.read_text(encoding="utf-8"))
            self.assertEqual(record["feedback"]["label"], "false_positive")
            self.assertEqual(record["event"]["evidence_uri"], event.evidence_uri)
            service.close()


if __name__ == "__main__":
    unittest.main()
