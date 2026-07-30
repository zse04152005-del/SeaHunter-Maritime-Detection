"""Bounded-memory flattened Parquet output for offline evaluation."""

from __future__ import annotations

import importlib
from datetime import timezone
from pathlib import Path
from typing import Any

from .results import FrameResult


class ParquetResultSink:
    """Write one row per detection and one null-detection row for empty frames."""

    def __init__(self, path: Path, *, batch_size: int = 256, compression: str = "zstd") -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not compression.strip():
            raise ValueError("compression must not be empty")
        self.path = path
        self.batch_size = batch_size
        self.compression = compression
        self._rows: list[dict[str, object]] = []
        self._writer: Any | None = None
        self._pa: Any | None = None
        self._pq: Any | None = None
        self._schema: Any | None = None
        self._closed = False
        self.frames_written = 0
        self.rows_written = 0

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("Parquet sink is closed")
        base: dict[str, object] = {
            "schema_version": 1,
            "source_id": result.frame.source_id,
            "frame_id": result.frame.frame_id,
            "captured_at_utc": result.frame.captured_at.astimezone(timezone.utc).replace(tzinfo=None),
            "source_pts_seconds": result.frame.source_pts_seconds,
            "width": result.frame.width,
            "height": result.frame.height,
            "dropped_before": result.dropped_before,
            "detector_id": result.detector_id,
            "frame_detection_count": len(result.detections),
        }
        if result.detections:
            ordered = sorted(
                result.detections,
                key=lambda detection: (
                    detection.class_id,
                    detection.bbox_xyxy,
                    -detection.confidence,
                    detection.detector_id,
                ),
            )
            for index, detection in enumerate(ordered):
                x1, y1, x2, y2 = detection.bbox_xyxy
                self._rows.append(
                    {
                        **base,
                        "detection_index": index,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "roi_id": detection.roi_id,
                    }
                )
        else:
            self._rows.append(
                {
                    **base,
                    "detection_index": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                    "bbox_x2": None,
                    "bbox_y2": None,
                    "class_id": None,
                    "class_name": None,
                    "confidence": None,
                    "roi_id": None,
                }
            )
        self.frames_written += 1
        if len(self._rows) >= self.batch_size:
            self._flush()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._closed = True

    def _flush(self) -> None:
        if not self._rows:
            return
        self._ensure_backend()
        assert self._pa is not None and self._pq is not None and self._schema is not None
        table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self._pq.ParquetWriter(
                str(self.path),
                self._schema,
                compression=self.compression,
                use_dictionary=True,
            )
        self._writer.write_table(table)
        self.rows_written += len(self._rows)
        self._rows.clear()

    def _ensure_backend(self) -> None:
        if self._pa is not None:
            return
        try:
            self._pa = importlib.import_module("pyarrow")
            self._pq = importlib.import_module("pyarrow.parquet")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Parquet output requires the 'parquet' project extra") from exc
        self._schema = self._pa.schema(
            [
                ("schema_version", self._pa.int16()),
                ("source_id", self._pa.string()),
                ("frame_id", self._pa.int64()),
                ("captured_at_utc", self._pa.timestamp("us")),
                ("source_pts_seconds", self._pa.float64()),
                ("width", self._pa.int32()),
                ("height", self._pa.int32()),
                ("dropped_before", self._pa.int32()),
                ("detector_id", self._pa.string()),
                ("frame_detection_count", self._pa.int32()),
                ("detection_index", self._pa.int32()),
                ("bbox_x1", self._pa.float32()),
                ("bbox_y1", self._pa.float32()),
                ("bbox_x2", self._pa.float32()),
                ("bbox_y2", self._pa.float32()),
                ("class_id", self._pa.int32()),
                ("class_name", self._pa.string()),
                ("confidence", self._pa.float32()),
                ("roi_id", self._pa.string()),
            ]
        )
