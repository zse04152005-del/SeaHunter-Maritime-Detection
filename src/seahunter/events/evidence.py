"""Bounded pre/post event evidence rings and atomic ZIP bundles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from seahunter.schemas import ModelManifest, RiskEvent, TelemetryPacket, TrackState

from .persistence import risk_event_to_dict


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    captured_at: datetime
    jpeg: bytes
    tracks: tuple[TrackState, ...] = ()
    telemetry: TelemetryPacket | None = None
    model_manifests: tuple[ModelManifest, ...] = ()

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if not self.jpeg:
            raise ValueError("evidence JPEG must not be empty")


@dataclass(frozen=True, slots=True)
class EvidenceRecorderConfig:
    pre_event_seconds: float = 10.0
    post_event_seconds: float = 10.0
    maximum_ring_frames: int = 900

    def __post_init__(self) -> None:
        if self.pre_event_seconds < 0.0 or self.post_event_seconds < 0.0:
            raise ValueError("evidence pre/post durations must be non-negative")
        if self.maximum_ring_frames <= 0:
            raise ValueError("maximum_ring_frames must be positive")


@dataclass(slots=True)
class _ActiveEvidence:
    event: RiskEvent
    deadline: datetime
    frames: list[EvidenceFrame]


class EvidenceRecorder:
    def __init__(self, output_directory: Path, config: EvidenceRecorderConfig | None = None) -> None:
        self.output_directory = output_directory
        self.config = config or EvidenceRecorderConfig()
        self._ring: deque[EvidenceFrame] = deque(maxlen=self.config.maximum_ring_frames)
        self._active: dict[str, _ActiveEvidence] = {}
        self._last_captured_at: datetime | None = None
        self._closed = False

    def ingest(self, frame: EvidenceFrame) -> tuple[Path, ...]:
        if self._closed:
            raise RuntimeError("evidence recorder is closed")
        if self._last_captured_at is not None and frame.captured_at <= self._last_captured_at:
            raise ValueError("evidence frames must increase strictly")
        self._last_captured_at = frame.captured_at
        self._ring.append(frame)
        cutoff = frame.captured_at - timedelta(seconds=self.config.pre_event_seconds)
        while self._ring and self._ring[0].captured_at < cutoff:
            self._ring.popleft()
        completed: list[Path] = []
        for event_id, active in list(self._active.items()):
            if frame.captured_at <= active.deadline:
                active.frames.append(frame)
            if frame.captured_at >= active.deadline:
                completed.append(self._finalize(active))
                self._active.pop(event_id, None)
        return tuple(completed)

    def open_event(self, event: RiskEvent) -> Path:
        if self._closed:
            raise RuntimeError("evidence recorder is closed")
        if event.event_id in self._active:
            return self.bundle_path(event.event_id)
        cutoff = event.opened_at - timedelta(seconds=self.config.pre_event_seconds)
        pre_frames = [frame for frame in self._ring if cutoff <= frame.captured_at <= event.opened_at]
        self._active[event.event_id] = _ActiveEvidence(
            event=event,
            deadline=event.opened_at + timedelta(seconds=self.config.post_event_seconds),
            frames=pre_frames,
        )
        return self.bundle_path(event.event_id)

    def update_event(self, event: RiskEvent) -> None:
        """Keep the evidence manifest aligned with the latest event lifecycle state."""

        active = self._active.get(event.event_id)
        if active is not None:
            active.event = event

    def bundle_path(self, event_id: str) -> Path:
        return self.output_directory / f"{event_id}.evidence.zip"

    def close(self) -> tuple[Path, ...]:
        if self._closed:
            return ()
        self._closed = True
        completed = tuple(self._finalize(active) for active in self._active.values())
        self._active.clear()
        return completed

    def _finalize(self, active: _ActiveEvidence) -> Path:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        target = self.bundle_path(active.event.event_id)
        temporary = target.with_suffix(target.suffix + ".tmp")
        frames = _deduplicate_frames(active.frames)
        hashes: dict[str, str] = {}
        track_lines: list[str] = []
        telemetry_lines: list[str] = []
        model_manifests: dict[str, dict[str, Any]] = {}
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, frame in enumerate(frames):
                name = f"frames/{index:06d}.jpg"
                archive.writestr(name, frame.jpeg)
                hashes[name] = hashlib.sha256(frame.jpeg).hexdigest()
                for track in frame.tracks:
                    track_lines.append(
                        json.dumps(
                            {
                                "captured_at": frame.captured_at.isoformat(),
                                "track": _jsonable(asdict(track)),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                if frame.telemetry is not None:
                    telemetry_lines.append(
                        json.dumps(
                            {
                                "captured_at": frame.captured_at.isoformat(),
                                "telemetry": _jsonable(asdict(frame.telemetry)),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                for manifest in frame.model_manifests:
                    model_manifests[f"{manifest.model_id}:{manifest.version}"] = _jsonable(asdict(manifest))
            tracks_payload = ("\n".join(track_lines) + ("\n" if track_lines else "")).encode()
            telemetry_payload = ("\n".join(telemetry_lines) + ("\n" if telemetry_lines else "")).encode()
            models_payload = (json.dumps(model_manifests, sort_keys=True, indent=2) + "\n").encode()
            for name, payload in (
                ("tracks.jsonl", tracks_payload),
                ("telemetry.jsonl", telemetry_payload),
                ("models.json", models_payload),
            ):
                archive.writestr(name, payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            bundle_manifest = {
                "schema_version": 1,
                "event": risk_event_to_dict(active.event),
                "pre_event_seconds": self.config.pre_event_seconds,
                "post_event_seconds": self.config.post_event_seconds,
                "frame_count": len(frames),
                "first_frame_at": None if not frames else frames[0].captured_at.isoformat(),
                "last_frame_at": None if not frames else frames[-1].captured_at.isoformat(),
                "model_manifests": model_manifests,
                "artifact_sha256": hashes,
            }
            archive.writestr("manifest.json", json.dumps(bundle_manifest, sort_keys=True, indent=2) + "\n")
        temporary.replace(target)
        return target


def _deduplicate_frames(frames: list[EvidenceFrame]) -> list[EvidenceFrame]:
    selected: dict[datetime, EvidenceFrame] = {}
    for frame in frames:
        selected[frame.captured_at] = frame
    return [selected[captured_at] for captured_at in sorted(selected)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
