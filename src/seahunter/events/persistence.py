"""SQLite event persistence and operator feedback audit."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock

from seahunter.schemas import EventStatus, RiskEvent


class FeedbackLabel(str, Enum):
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    UNSURE = "unsure"


@dataclass(frozen=True, slots=True)
class OperatorFeedback:
    event_id: str
    label: FeedbackLabel
    operator_id: str
    created_at: datetime
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.operator_id.strip():
            raise ValueError("event_id and operator_id must not be empty")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.note is not None and not self.note.strip():
            raise ValueError("feedback note must not be empty")


class SQLiteEventStore:
    """Durable latest-event state plus append-only transitions and feedback."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_transitions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operator_feedback (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                label TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                note TEXT
            );
            """
        )
        self._connection.commit()

    def append(self, event: RiskEvent) -> None:
        payload = json.dumps(risk_event_to_dict(event), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO events(event_id, payload_json, status, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (event.event_id, payload, event.status.value, event.updated_at.isoformat()),
            )
            self._connection.execute(
                "INSERT INTO event_transitions(event_id, payload_json, recorded_at) VALUES (?, ?, ?)",
                (event.event_id, payload, event.updated_at.isoformat()),
            )

    def get(self, event_id: str) -> RiskEvent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return None if row is None else risk_event_from_dict(json.loads(row["payload_json"]))

    def list_events(self, *, active_only: bool = False) -> tuple[RiskEvent, ...]:
        query = "SELECT payload_json FROM events"
        parameters: tuple[str, ...] = ()
        if active_only:
            query += " WHERE status != ?"
            parameters = (EventStatus.CLOSED.value,)
        query += " ORDER BY updated_at, event_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(risk_event_from_dict(json.loads(row["payload_json"])) for row in rows)

    def list_transitions(self, event_id: str) -> tuple[RiskEvent, ...]:
        """Return the immutable transition history for one event."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM event_transitions WHERE event_id = ? ORDER BY sequence",
                (event_id,),
            ).fetchall()
        return tuple(risk_event_from_dict(json.loads(row["payload_json"])) for row in rows)

    def add_feedback(self, feedback: OperatorFeedback) -> None:
        if self.get(feedback.event_id) is None:
            raise KeyError("feedback event_id not found")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO operator_feedback(event_id, label, operator_id, created_at, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback.event_id,
                    feedback.label.value,
                    feedback.operator_id,
                    feedback.created_at.isoformat(),
                    feedback.note,
                ),
            )

    def list_feedback(self, event_id: str | None = None) -> tuple[OperatorFeedback, ...]:
        query = "SELECT event_id, label, operator_id, created_at, note FROM operator_feedback"
        parameters: tuple[str, ...] = ()
        if event_id is not None:
            query += " WHERE event_id = ?"
            parameters = (event_id,)
        query += " ORDER BY sequence"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(
            OperatorFeedback(
                event_id=str(row["event_id"]),
                label=FeedbackLabel(str(row["label"])),
                operator_id=str(row["operator_id"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                note=None if row["note"] is None else str(row["note"]),
            )
            for row in rows
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def risk_event_to_dict(event: RiskEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "rule_id": event.rule_id,
        "zone_id": event.zone_id,
        "track_id": event.track_id,
        "opened_at": event.opened_at.isoformat(),
        "updated_at": event.updated_at.isoformat(),
        "status": event.status.value,
        "risk_score": event.risk_score,
        "evidence_uri": event.evidence_uri,
    }


def risk_event_from_dict(payload: dict[str, object]) -> RiskEvent:
    return RiskEvent(
        event_id=str(payload["event_id"]),
        rule_id=str(payload["rule_id"]),
        zone_id=str(payload["zone_id"]),
        track_id=int(str(payload["track_id"])),
        opened_at=datetime.fromisoformat(str(payload["opened_at"])),
        updated_at=datetime.fromisoformat(str(payload["updated_at"])),
        status=EventStatus(str(payload["status"])),
        risk_score=float(str(payload["risk_score"])),
        evidence_uri=None if payload.get("evidence_uri") is None else str(payload["evidence_uri"]),
    )
