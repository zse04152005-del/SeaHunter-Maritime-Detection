from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from seahunter.events import (
    DangerZone,
    DangerZoneEngine,
    DangerZoneEngineConfig,
    EventService,
    EvidenceRecorder,
    HardSampleFeedbackWriter,
    SQLiteEventStore,
    ZoneCoordinateSpace,
    ZoneObservation,
    create_event_app,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def make_engine() -> DangerZoneEngine:
    zone = DangerZone(
        zone_id="restricted",
        name="Restricted",
        polygon=((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
        coordinate_space=ZoneCoordinateSpace.LOCAL_ENU,
        severity=0.9,
    )
    return DangerZoneEngine((zone,), DangerZoneEngineConfig(minimum_enter_seconds=0.0, minimum_exit_seconds=0.0))


def make_observation(seconds: float) -> ZoneObservation:
    return ZoneObservation(
        track_id=8,
        captured_at=START + timedelta(seconds=seconds),
        point=(10.0, 10.0),
        velocity_en_m_s=(0.0, 0.0),
        reliability=0.95,
        class_id=1,
    )


class EventAPITests(unittest.TestCase):
    def test_rest_acknowledgement_feedback_and_websocket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EventService(
                make_engine(),
                SQLiteEventStore(root / "events.sqlite3"),
                EvidenceRecorder(root / "evidence"),
                hard_samples=HardSampleFeedbackWriter(root / "hard-samples.jsonl"),
            )
            opened = next(item for item in service.update(make_observation(0.0)) if item.rule_id == "enter")

            with TestClient(create_event_app(service)) as client:
                self.assertEqual(client.get("/health").json()["active_events"], 1)
                self.assertEqual(client.get("/events?active_only=true").json()[0]["event_id"], opened.event_id)
                self.assertEqual(client.get(f"/events/{opened.event_id}").status_code, 200)
                self.assertEqual(client.get("/events/missing").status_code, 404)

                acknowledged = client.post(
                    f"/events/{opened.event_id}/acknowledge",
                    json={"acknowledged_at": "2026-07-30T00:00:01Z"},
                )
                self.assertEqual(acknowledged.status_code, 200)
                self.assertEqual(acknowledged.json()["status"], "acknowledged")
                transitions = client.get(f"/events/{opened.event_id}/transitions").json()
                self.assertEqual([item["status"] for item in transitions], ["open", "acknowledged"])

                feedback = client.post(
                    f"/events/{opened.event_id}/feedback",
                    json={
                        "label": "false_positive",
                        "operator_id": "operator-1",
                        "created_at": "2026-07-30T00:00:02+00:00",
                        "note": "foam wake",
                    },
                )
                self.assertEqual(feedback.status_code, 200)
                self.assertEqual(feedback.json()["label"], "false_positive")

                with client.websocket_connect("/ws/events") as websocket:
                    first = websocket.receive_json()
                    second = websocket.receive_json()
                self.assertEqual(first["event"]["status"], "open")
                self.assertEqual(second["event"]["status"], "acknowledged")

            hard_sample = json.loads((root / "hard-samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(hard_sample["event"]["event_id"], opened.event_id)
            service.close()

    def test_invalid_feedback_and_acknowledgement_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = EventService(
                make_engine(), SQLiteEventStore(root / "events.sqlite3"), EvidenceRecorder(root / "evidence")
            )
            opened = next(item for item in service.update(make_observation(0.0)) if item.rule_id == "enter")
            with TestClient(create_event_app(service)) as client:
                self.assertEqual(
                    client.post(f"/events/{opened.event_id}/feedback", json={"operator_id": "op"}).status_code,
                    422,
                )
                self.assertEqual(
                    client.post(
                        f"/events/{opened.event_id}/acknowledge",
                        json={"acknowledged_at": "2026-07-30T00:00:01"},
                    ).status_code,
                    422,
                )
            service.close()


if __name__ == "__main__":
    unittest.main()
