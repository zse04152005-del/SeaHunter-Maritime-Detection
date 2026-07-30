"""Append-only operator feedback export for active-learning hard samples."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from seahunter.schemas import RiskEvent

from .persistence import OperatorFeedback, risk_event_to_dict


class HardSampleFeedbackWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def write(self, event: RiskEvent, feedback: OperatorFeedback) -> None:
        record = {
            "schema_version": 1,
            "event": risk_event_to_dict(event),
            "feedback": {
                "event_id": feedback.event_id,
                "label": feedback.label.value,
                "operator_id": feedback.operator_id,
                "created_at": feedback.created_at.isoformat(),
                "note": feedback.note,
            },
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
