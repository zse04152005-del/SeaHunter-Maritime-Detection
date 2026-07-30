"""OpenCV annotation, video recording, and JPEG preview result sinks."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .preview import PreviewHub
from .results import FrameResult, frame_result_to_record

_COLORS = (
    (80, 180, 255),
    (80, 220, 120),
    (255, 160, 80),
    (220, 100, 220),
    (80, 220, 220),
    (255, 100, 100),
)


def render_annotated_frame(result: FrameResult) -> Any:
    """Copy a decoded BGR frame and draw detections plus timing status."""

    payload = result.frame.payload
    if payload is None or not hasattr(payload, "copy"):
        raise ValueError("FrameResult.frame.payload must be an image array")
    cv2: Any = importlib.import_module("cv2")
    image = payload.copy()
    height, width = image.shape[:2]
    line_width = max(1, round(min(width, height) / 400))
    font_scale = max(0.4, min(width, height) / 900)

    for detection in result.detections:
        color = _COLORS[detection.class_id % len(_COLORS)]
        x1, y1, x2, y2 = detection.bbox_xyxy
        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(0, min(width - 1, round(x2)))
        bottom = max(0, min(height - 1, round(y2)))
        cv2.rectangle(image, (left, top), (right, bottom), color, line_width)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            line_width,
        )
        label_top = max(0, top - text_height - baseline - 4)
        cv2.rectangle(
            image,
            (left, label_top),
            (min(width - 1, left + text_width + 6), top),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (left + 3, max(text_height, top - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 20, 30),
            line_width,
            cv2.LINE_AA,
        )

    status = (
        f"{result.frame.source_id} frame={result.frame.frame_id} "
        f"objects={len(result.detections)} infer={result.inference_duration_ms:.1f}ms "
        f"dropped={result.dropped_before}"
    )
    cv2.putText(
        image,
        status,
        (10, max(20, round(28 * font_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, line_width),
        cv2.LINE_AA,
    )
    return image


class AnnotatedVideoSink:
    """Stream annotated frames into a bounded-memory video writer."""

    def __init__(self, path: Path, *, fps: float = 25.0, codec: str | None = None) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if codec is not None and len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        self.path = path
        self.fps = fps
        self.codec = codec or ("MJPG" if path.suffix.lower() == ".avi" else "mp4v")
        self._writer: Any | None = None
        self._dimensions: tuple[int, int] | None = None
        self._closed = False
        self.frames_written = 0

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("annotated video sink is closed")
        cv2: Any = importlib.import_module("cv2")
        image = render_annotated_frame(result)
        height, width = image.shape[:2]
        dimensions = (width, height)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*self.codec),
                self.fps,
                dimensions,
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"failed to open annotated video writer: {self.path}")
            self._dimensions = dimensions
        elif dimensions != self._dimensions:
            raise ValueError(f"annotated video frame size changed from {self._dimensions} to {dimensions}")
        self._writer.write(image)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class JpegPreviewSink:
    """Publish the latest annotated JPEG without retaining historical frames."""

    def __init__(self, hub: PreviewHub, *, quality: int = 80, close_hub_on_close: bool = True) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("quality must be within [1, 100]")
        self.hub = hub
        self.quality = quality
        self.close_hub_on_close = close_hub_on_close
        self._closed = False
        self.frames_published = 0

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("JPEG preview sink is closed")
        cv2: Any = importlib.import_module("cv2")
        image = render_annotated_frame(result)
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if not success:
            raise RuntimeError("failed to encode preview JPEG")
        self.hub.publish(encoded.tobytes(), frame_result_to_record(result))
        self.frames_published += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_hub_on_close:
            self.hub.close()
