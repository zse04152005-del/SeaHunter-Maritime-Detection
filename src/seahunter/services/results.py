"""Normalized per-frame inference results and output sink contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Protocol

from seahunter.schemas import Detection, FramePacket


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One fully normalized detector result with runtime context."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    detector_id: str
    dropped_before: int
    inference_duration_ms: float

    def __post_init__(self) -> None:
        if not self.detector_id.strip():
            raise ValueError("detector_id must not be empty")
        if self.dropped_before < 0:
            raise ValueError("dropped_before must be non-negative")
        if not isfinite(self.inference_duration_ms) or self.inference_duration_ms < 0:
            raise ValueError("inference_duration_ms must be finite and non-negative")


class FrameResultSink(Protocol):
    """Consumer for video, preview, database, or message-bus outputs."""

    def write(self, result: FrameResult) -> None: ...

    def close(self) -> None: ...


def frame_result_to_record(result: FrameResult, *, include_runtime_timings: bool = False) -> dict[str, object]:
    """Convert a frame result into the canonical versioned metadata record."""

    ordered = sorted(
        result.detections,
        key=lambda detection: (
            detection.class_id,
            detection.bbox_xyxy,
            -detection.confidence,
            detection.detector_id,
        ),
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "source_id": result.frame.source_id,
        "frame_id": result.frame.frame_id,
        "captured_at": _utc_isoformat(result.frame.captured_at),
        "source_pts_seconds": (
            None if result.frame.source_pts_seconds is None else round(result.frame.source_pts_seconds, 6)
        ),
        "width": result.frame.width,
        "height": result.frame.height,
        "dropped_before": result.dropped_before,
        "detector_id": result.detector_id,
        "detections": [
            {
                "bbox_xyxy": [round(value, 6) for value in detection.bbox_xyxy],
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": round(detection.confidence, 6),
                "detector_id": detection.detector_id,
                "roi_id": detection.roi_id,
            }
            for detection in ordered
        ],
    }
    if include_runtime_timings:
        record["runtime_timing_ms"] = {
            "decode": (None if result.frame.decode_duration_ms is None else round(result.frame.decode_duration_ms, 6)),
            "inference": round(result.inference_duration_ms, 6),
        }
    return record


def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
