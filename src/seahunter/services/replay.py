"""Deterministic video replay with separate runtime performance reporting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol

from seahunter.perception import Detector
from seahunter.schemas import Detection, FramePacket
from seahunter.video import BufferClosed, FrameIngestWorker, ReaderStats, VideoSource


class ReplayFrameReader(Protocol):
    """Reader capabilities required by the replay service."""

    source: VideoSource

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...

    def stats(self) -> ReaderStats: ...


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """End-to-end replay counters and latency percentiles."""

    schema_version: int
    source_id: str
    detector_id: str
    mode: str
    frames_processed: int
    detections_emitted: int
    source_frames_decoded: int
    queue_frames_dropped: int
    reader_read_failures: int
    reconnect_attempts: int
    wall_duration_seconds: float
    throughput_fps: float
    decode_p50_ms: float
    decode_p95_ms: float
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    metadata_sha256: str

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serializable summary."""

        return asdict(self)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def run_replay(
    reader: ReplayFrameReader,
    detector: Detector,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    max_frames: int | None = None,
    realtime: bool = False,
    buffer_capacity: int = 2,
    include_runtime_timings: bool = False,
) -> ReplaySummary:
    """Run detection over a source and write canonical JSONL frame records.

    The default JSONL excludes wall-clock inference timing, so identical detector
    outputs produce byte-stable metadata. Runtime timing is always available in the
    separate summary and can be embedded per frame only through an explicit flag.
    """

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    if buffer_capacity <= 0:
        raise ValueError("buffer_capacity must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    inference_latencies: list[float] = []
    decode_latencies: list[float] = []
    frames_processed = 0
    detections_emitted = 0
    queue_frames_dropped = 0
    last_frame_id: int | None = None
    source_id = reader.source.suggested_id()
    worker: FrameIngestWorker | None = None
    started = perf_counter()

    try:
        if realtime:
            worker = FrameIngestWorker(reader, capacity=buffer_capacity)
            worker.start()
            frames = _worker_frames(worker)
        else:
            frames = _reader_frames(reader)

        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for frame in frames:
                source_id = frame.source_id
                inferred_drop = 0 if last_frame_id is None else max(0, frame.frame_id - last_frame_id - 1)
                dropped_before = max(frame.dropped_before, inferred_drop)
                last_frame_id = frame.frame_id

                inference_started = perf_counter()
                detections = list(detector.infer(frame))
                inference_ms = max(0.0, (perf_counter() - inference_started) * 1000.0)
                inference_latencies.append(inference_ms)
                if frame.decode_duration_ms is not None:
                    decode_latencies.append(frame.decode_duration_ms)

                record = _frame_record(
                    frame,
                    detections,
                    detector.detector_id,
                    dropped_before=dropped_before,
                    inference_ms=inference_ms if include_runtime_timings else None,
                )
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
                frames_processed += 1
                detections_emitted += len(detections)
                if max_frames is not None and frames_processed >= max_frames:
                    break
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=5.0)
            queue_frames_dropped = worker.stats().buffer.dropped
        else:
            reader.close()

    wall_duration = max(0.0, perf_counter() - started)
    reader_stats = reader.stats()
    metadata_sha256 = _sha256_file(output_path)
    summary = ReplaySummary(
        schema_version=1,
        source_id=source_id,
        detector_id=detector.detector_id,
        mode="realtime" if realtime else "offline",
        frames_processed=frames_processed,
        detections_emitted=detections_emitted,
        source_frames_decoded=reader_stats.frames_decoded,
        queue_frames_dropped=queue_frames_dropped,
        reader_read_failures=reader_stats.read_failures,
        reconnect_attempts=reader_stats.reconnect_attempts,
        wall_duration_seconds=round(wall_duration, 6),
        throughput_fps=round(frames_processed / wall_duration, 6) if wall_duration > 0 else 0.0,
        decode_p50_ms=round(percentile(decode_latencies, 0.50), 6),
        decode_p95_ms=round(percentile(decode_latencies, 0.95), 6),
        inference_p50_ms=round(percentile(inference_latencies, 0.50), 6),
        inference_p95_ms=round(percentile(inference_latencies, 0.95), 6),
        inference_p99_ms=round(percentile(inference_latencies, 0.99), 6),
        metadata_sha256=metadata_sha256,
    )
    if summary_path is not None:
        summary_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def _reader_frames(reader: ReplayFrameReader) -> Iterator[FramePacket]:
    while True:
        frame = reader.read()
        if frame is None:
            return
        yield frame


def _worker_frames(worker: FrameIngestWorker) -> Iterator[FramePacket]:
    while True:
        try:
            yield worker.get()
        except BufferClosed:
            return


def _frame_record(
    frame: FramePacket,
    detections: Sequence[Detection],
    detector_id: str,
    *,
    dropped_before: int,
    inference_ms: float | None,
) -> dict[str, object]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            detection.class_id,
            detection.bbox_xyxy,
            -detection.confidence,
            detection.detector_id,
        ),
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "source_id": frame.source_id,
        "frame_id": frame.frame_id,
        "captured_at": _utc_isoformat(frame.captured_at),
        "source_pts_seconds": None if frame.source_pts_seconds is None else round(frame.source_pts_seconds, 6),
        "width": frame.width,
        "height": frame.height,
        "dropped_before": dropped_before,
        "detector_id": detector_id,
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
    if inference_ms is not None:
        record["runtime_timing_ms"] = {
            "decode": None if frame.decode_duration_ms is None else round(frame.decode_duration_ms, 6),
            "inference": round(inference_ms, 6),
        }
    return record


def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
