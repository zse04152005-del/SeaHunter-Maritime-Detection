"""Dependency-light MAVLink message aggregation into SeaHunter telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import degrees, isfinite
from typing import Any

from seahunter.schemas import AltitudeDatum, TelemetryPacket

from .time_sync import ClockSynchronizer


@dataclass(slots=True)
class _MAVState:
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float | None = None
    altitude_datum: AltitudeDatum = AltitudeDatum.UNKNOWN
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    gimbal_roll_deg: float = 0.0
    gimbal_pitch_deg: float = 0.0
    gimbal_yaw_deg: float = 0.0


class MAVLinkTelemetryAdapter:
    """Aggregate common pymavlink/dict messages after clock acceptance."""

    def __init__(self, source_id: str, *, clock: ClockSynchronizer | None = None) -> None:
        if not source_id.strip():
            raise ValueError("source_id must not be empty")
        self.source_id = source_id
        self.clock = clock or ClockSynchronizer()
        self._state = _MAVState()

    def ingest(self, message: Any, received_at: datetime) -> TelemetryPacket | None:
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")
        message_type = _message_type(message)
        boot_ms = _optional_number(message, "time_boot_ms")
        if boot_ms is None:
            return None
        source_seconds = boot_ms / 1_000.0
        sync = self.clock.add_sample(source_seconds, received_at)

        if message_type == "GLOBAL_POSITION_INT":
            latitude = _required_number(message, "lat") / 10_000_000.0
            longitude = _required_number(message, "lon") / 10_000_000.0
            relative_altitude = _optional_number(message, "relative_alt")
            altitude = _required_number(message, "alt") if relative_altitude is None else relative_altitude
            self._state.latitude_deg = latitude
            self._state.longitude_deg = longitude
            self._state.altitude_m = altitude / 1_000.0
            self._state.altitude_datum = (
                AltitudeDatum.AMSL if relative_altitude is None else AltitudeDatum.RELATIVE_HOME
            )
        elif message_type == "ATTITUDE":
            self._state.roll_deg = degrees(_required_number(message, "roll"))
            self._state.pitch_deg = degrees(_required_number(message, "pitch"))
            self._state.yaw_deg = degrees(_required_number(message, "yaw"))
        elif message_type == "MOUNT_ORIENTATION":
            self._state.gimbal_roll_deg = _required_number(message, "roll")
            self._state.gimbal_pitch_deg = _required_number(message, "pitch")
            self._state.gimbal_yaw_deg = _required_number(message, "yaw")
        else:
            return None

        captured_at = self.clock.align(source_seconds)
        state = self._state
        if (
            not sync.accepted
            or captured_at is None
            or state.latitude_deg is None
            or state.longitude_deg is None
            or state.altitude_m is None
            or state.roll_deg is None
            or state.pitch_deg is None
            or state.yaw_deg is None
        ):
            return None
        return TelemetryPacket(
            source_id=self.source_id,
            captured_at=captured_at,
            latitude_deg=state.latitude_deg,
            longitude_deg=state.longitude_deg,
            altitude_m=state.altitude_m,
            platform_roll_deg=state.roll_deg,
            platform_pitch_deg=state.pitch_deg,
            platform_yaw_deg=state.yaw_deg,
            gimbal_roll_deg=state.gimbal_roll_deg,
            gimbal_pitch_deg=state.gimbal_pitch_deg,
            gimbal_yaw_deg=state.gimbal_yaw_deg,
            quality=sync.quality,
            altitude_datum=state.altitude_datum,
        )


def _message_type(message: Any) -> str:
    if isinstance(message, Mapping):
        value = message.get("mavpackettype", message.get("type"))
    else:
        getter = getattr(message, "get_type", None)
        value = getter() if callable(getter) else getattr(message, "mavpackettype", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MAVLink message type is missing")
    return value.strip().upper()


def _optional_number(message: Any, key: str) -> float | None:
    value = message.get(key) if isinstance(message, Mapping) else getattr(message, key, None)
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"MAVLink field {key} must be finite")
    return numeric


def _required_number(message: Any, key: str) -> float:
    value = _optional_number(message, key)
    if value is None:
        raise ValueError(f"MAVLink field {key} is missing")
    return value
