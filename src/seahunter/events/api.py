"""FastAPI REST and WebSocket interface for durable risk events."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone
from typing import Any

from .persistence import FeedbackLabel, OperatorFeedback, risk_event_to_dict
from .service import EventService


def create_event_app(service: EventService) -> Any:
    """Create an edge-safe event API without making FastAPI a core dependency."""

    try:
        fastapi: Any = importlib.import_module("fastapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError("event APIs require the 'edge' project extra") from exc

    app = fastapi.FastAPI(title="SeaHunter-VIS event service", version="0.1.0")

    async def health() -> dict[str, object]:
        return {
            "status": "stopped" if service.hub.closed else "running",
            "active_events": len(service.list_events(active_only=True)),
            "mqtt_publication_failures": len(service.publication_failures()),
        }

    async def list_events(active_only: bool = False) -> list[dict[str, object]]:
        return [risk_event_to_dict(event) for event in service.list_events(active_only=active_only)]

    async def get_event(event_id: str) -> dict[str, object]:
        event = service.get_event(event_id)
        if event is None:
            raise fastapi.HTTPException(status_code=404, detail="event not found")
        return risk_event_to_dict(event)

    async def get_transitions(event_id: str) -> list[dict[str, object]]:
        if service.get_event(event_id) is None:
            raise fastapi.HTTPException(status_code=404, detail="event not found")
        return [risk_event_to_dict(event) for event in service.list_transitions(event_id)]

    async def acknowledge(event_id: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        try:
            acknowledged_at = _parse_timestamp(None if payload is None else payload.get("acknowledged_at"))
            event = service.acknowledge(event_id, acknowledged_at)
        except KeyError as exc:
            raise fastapi.HTTPException(status_code=404, detail="active event not found") from exc
        except (TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=422, detail=str(exc)) from exc
        return risk_event_to_dict(event)

    async def add_feedback(event_id: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            if "label" not in payload or "operator_id" not in payload:
                raise ValueError("label and operator_id are required")
            note_value = payload.get("note")
            feedback = OperatorFeedback(
                event_id=event_id,
                label=FeedbackLabel(str(payload["label"])),
                operator_id=str(payload["operator_id"]),
                created_at=_parse_timestamp(payload.get("created_at")),
                note=None if note_value is None else str(note_value),
            )
            service.add_feedback(feedback)
        except KeyError as exc:
            raise fastapi.HTTPException(status_code=404, detail="event not found") from exc
        except (TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "event_id": feedback.event_id,
            "label": feedback.label.value,
            "operator_id": feedback.operator_id,
            "created_at": feedback.created_at.isoformat(),
            "note": feedback.note,
        }

    async def events_websocket(websocket: Any) -> None:
        await websocket.accept()
        after_sequence = -1
        try:
            while True:
                published = await asyncio.to_thread(service.hub.wait_for_next, after_sequence, 0.25)
                if published is None:
                    if service.hub.closed:
                        await websocket.close(code=1000)
                        return
                    continue
                after_sequence = published.sequence
                await websocket.send_json(
                    {
                        "type": "event",
                        "sequence": published.sequence,
                        "event": risk_event_to_dict(published.event),
                    }
                )
        except fastapi.WebSocketDisconnect:
            return

    events_websocket.__annotations__["websocket"] = fastapi.WebSocket
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/events", list_events, methods=["GET"])
    app.add_api_route("/events/{event_id}", get_event, methods=["GET"])
    app.add_api_route("/events/{event_id}/transitions", get_transitions, methods=["GET"])
    app.add_api_route("/events/{event_id}/acknowledge", acknowledge, methods=["POST"])
    app.add_api_route("/events/{event_id}/feedback", add_feedback, methods=["POST"])
    app.add_api_websocket_route("/ws/events", events_websocket)
    return app


def _parse_timestamp(value: object) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, str):
        raise TypeError("timestamp must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed
