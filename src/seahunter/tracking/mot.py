"""Streaming MOTChallenge and auditable JSONL track-state writers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timezone
from pathlib import Path
from typing import TextIO

from seahunter.schemas import ObservationKind, TrackState


class MOTChallengeWriter:
    """Write standard ten-column MOTChallenge rows without buffering history."""

    def __init__(self, path: Path, *, include_inferred: bool = False, frame_id_offset: int = 1) -> None:
        if frame_id_offset < 0:
            raise ValueError("frame_id_offset must be non-negative")
        self.path = path
        self.include_inferred = include_inferred
        self.frame_id_offset = frame_id_offset
        self._stream: TextIO | None = None
        self._closed = False
        self.rows_written = 0

    def write(self, states: Sequence[TrackState]) -> None:
        if self._closed:
            raise RuntimeError("MOT writer is closed")
        selected = [state for state in states if self.include_inferred or state.observation is ObservationKind.OBSERVED]
        if not selected:
            return
        stream = self._ensure_stream()
        for state in sorted(selected, key=lambda item: (item.frame_id, item.track_id)):
            x1, y1, x2, y2 = state.bbox_xyxy
            row = (
                f"{state.frame_id + self.frame_id_offset},{state.track_id},"
                f"{x1:.6f},{y1:.6f},{max(0.0, x2 - x1):.6f},{max(0.0, y2 - y1):.6f},"
                f"{state.confidence:.6f},-1,-1,-1\n"
            )
            stream.write(row)
            self.rows_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        elif not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _ensure_stream(self) -> TextIO:
        if self._stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", newline="\n")
        return self._stream


class TrackJsonlWriter:
    """Write complete track states including inferred-point audit fields."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: TextIO | None = None
        self._closed = False
        self.rows_written = 0

    def write(self, states: Sequence[TrackState]) -> None:
        if self._closed:
            raise RuntimeError("track JSONL writer is closed")
        if not states:
            return
        stream = self._ensure_stream()
        for state in sorted(states, key=lambda item: (item.frame_id, item.track_id)):
            record = {
                "schema_version": 5,
                "tracker_id": state.tracker_id,
                "track_id": state.track_id,
                "frame_id": state.frame_id,
                "captured_at": state.captured_at.astimezone(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                "bbox_xyxy": [round(value, 6) for value in state.bbox_xyxy],
                "class_id": state.class_id,
                "confidence": round(state.confidence, 6),
                "observation": state.observation.value,
                "lifecycle": state.lifecycle.value,
                "velocity_xy_px_s": [round(value, 6) for value in state.velocity_xy_px_s],
                "covariance_xy": [round(value, 6) for value in state.covariance_xy],
                "age_frames": state.age_frames,
                "time_since_update": state.time_since_update,
                "association_score": (None if state.association_score is None else round(state.association_score, 6)),
                "association_stage": state.association_stage,
                "motion_gate_distance": (
                    None if state.motion_gate_distance is None else round(state.motion_gate_distance, 6)
                ),
                "appearance_score": None if state.appearance_score is None else round(state.appearance_score, 6),
                "reid_eligible": state.reid_eligible,
                "reid_bypass_reason": state.reid_bypass_reason,
                "lost_reason": None if state.lost_reason is None else state.lost_reason.value,
                "inference_method": None if state.inference_method is None else state.inference_method.value,
                "global_motion": (
                    None
                    if state.global_motion_affine is None
                    else {
                        "affine_2x3": [round(value, 6) for value in state.global_motion_affine],
                        "quality": round(state.global_motion_quality or 0.0, 6),
                        "applied": state.global_motion_applied,
                        "fallback_reason": state.global_motion_fallback_reason,
                        "source": state.global_motion_source,
                        "visual_quality": (
                            None
                            if state.global_motion_visual_quality is None
                            else round(state.global_motion_visual_quality, 6)
                        ),
                        "prior_quality": (
                            None
                            if state.global_motion_prior_quality is None
                            else round(state.global_motion_prior_quality, 6)
                        ),
                        "fusion_reason": state.global_motion_fusion_reason,
                    }
                ),
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            self.rows_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        elif not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _ensure_stream(self) -> TextIO:
        if self._stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", newline="\n")
        return self._stream
