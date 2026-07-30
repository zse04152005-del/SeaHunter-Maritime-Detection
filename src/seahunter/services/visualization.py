"""OpenCV annotation, video recording, and JPEG preview result sinks."""

from __future__ import annotations

import importlib
from collections import deque
from collections.abc import Callable, Sequence
from math import hypot
from pathlib import Path
from typing import Any

from seahunter.schemas import ObservationKind, TrackState

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
_INFERRED_COLOR = (0, 165, 255)

TrackStateProvider = Callable[[], Sequence[TrackState]]


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


class TrackOverlayRenderer:
    """Render bounded track trails with explicit observed/inferred styling."""

    def __init__(self, *, trail_length: int = 30) -> None:
        if trail_length < 0:
            raise ValueError("trail_length must be non-negative")
        self.trail_length = trail_length
        self._trails: dict[int, deque[tuple[tuple[int, int], ObservationKind]]] = {}
        self._source_id: str | None = None
        self._last_frame_id: int | None = None

    def render(self, result: FrameResult, states: Sequence[TrackState]) -> Any:
        """Copy a decoded frame and draw current track states plus bounded trails."""

        payload = result.frame.payload
        if payload is None or not hasattr(payload, "copy"):
            raise ValueError("FrameResult.frame.payload must be an image array")
        ordered_states = tuple(sorted(states, key=lambda state: state.track_id))
        for state in ordered_states:
            if state.frame_id != result.frame.frame_id:
                raise ValueError("track state frame_id must match the rendered FrameResult")

        if self._source_id != result.frame.source_id or (
            self._last_frame_id is not None and result.frame.frame_id <= self._last_frame_id
        ):
            self.reset()
        self._source_id = result.frame.source_id
        self._last_frame_id = result.frame.frame_id

        cv2: Any = importlib.import_module("cv2")
        image = payload.copy()
        height, width = image.shape[:2]
        line_width = max(1, round(min(width, height) / 400))
        font_scale = max(0.4, min(width, height) / 900)

        active_ids = {state.track_id for state in ordered_states}
        for track_id in tuple(self._trails):
            if track_id not in active_ids:
                del self._trails[track_id]

        if self.trail_length > 0:
            for state in ordered_states:
                center = _bbox_center(state.bbox_xyxy, width=width, height=height)
                trail = self._trails.setdefault(state.track_id, deque(maxlen=self.trail_length))
                trail.append((center, state.observation))

        for state in ordered_states:
            identity_color = _COLORS[state.track_id % len(_COLORS)]
            track_trail = self._trails.get(state.track_id)
            previous: tuple[int, int] | None = None
            if track_trail is not None:
                for point, observation in track_trail:
                    if previous is not None:
                        if observation is ObservationKind.INFERRED:
                            _draw_dashed_line(cv2, image, previous, point, _INFERRED_COLOR, line_width)
                        else:
                            cv2.line(image, previous, point, identity_color, line_width, cv2.LINE_AA)
                    previous = point

            left, top, right, bottom = _clamp_bbox(state.bbox_xyxy, width=width, height=height)
            if state.observation is ObservationKind.INFERRED:
                display_color = _INFERRED_COLOR
                _draw_dashed_rectangle(
                    cv2,
                    image,
                    (left, top),
                    (right, bottom),
                    display_color,
                    line_width,
                )
            else:
                display_color = identity_color
                cv2.rectangle(image, (left, top), (right, bottom), display_color, line_width)

            observation_label = "obs" if state.observation is ObservationKind.OBSERVED else "pred"
            label = (
                f"ID {state.track_id} c{state.class_id} {state.lifecycle.value}/{observation_label} "
                f"{state.confidence:.2f}"
            )
            _draw_label(
                cv2,
                image,
                label,
                left=left,
                top=top,
                width=width,
                color=display_color,
                font_scale=font_scale,
                line_width=line_width,
            )

        observed = sum(state.observation is ObservationKind.OBSERVED for state in ordered_states)
        inferred = len(ordered_states) - observed
        status = (
            f"{result.frame.source_id} frame={result.frame.frame_id} tracks={len(ordered_states)} "
            f"observed={observed} inferred={inferred} infer={result.inference_duration_ms:.1f}ms "
            f"dropped={result.dropped_before}"
        )
        cv2.putText(
            image,
            status,
            (10, max(20, round(28 * font_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
        return image

    def reset(self) -> None:
        """Clear trails when a source or sequence changes."""

        self._trails.clear()
        self._source_id = None
        self._last_frame_id = None


def render_tracked_frame(
    result: FrameResult,
    states: Sequence[TrackState],
    *,
    trail_length: int = 30,
) -> Any:
    """Render one tracked frame without retaining history between calls."""

    return TrackOverlayRenderer(trail_length=trail_length).render(result, states)


class AnnotatedVideoSink:
    """Stream annotated frames into a bounded-memory video writer."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float = 25.0,
        codec: str | None = None,
        track_state_provider: TrackStateProvider | None = None,
        track_trail_length: int = 30,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if codec is not None and len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        self.path = path
        self.fps = fps
        self.codec = codec or ("MJPG" if path.suffix.lower() == ".avi" else "mp4v")
        self._track_state_provider = track_state_provider
        self._track_renderer = (
            None if track_state_provider is None else TrackOverlayRenderer(trail_length=track_trail_length)
        )
        self._writer: Any | None = None
        self._dimensions: tuple[int, int] | None = None
        self._closed = False
        self.frames_written = 0

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("annotated video sink is closed")
        cv2: Any = importlib.import_module("cv2")
        image, _ = _render_result(result, self._track_state_provider, self._track_renderer)
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

    def __init__(
        self,
        hub: PreviewHub,
        *,
        quality: int = 80,
        close_hub_on_close: bool = True,
        track_state_provider: TrackStateProvider | None = None,
        track_trail_length: int = 30,
    ) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("quality must be within [1, 100]")
        self.hub = hub
        self.quality = quality
        self.close_hub_on_close = close_hub_on_close
        self._track_state_provider = track_state_provider
        self._track_renderer = (
            None if track_state_provider is None else TrackOverlayRenderer(trail_length=track_trail_length)
        )
        self._closed = False
        self.frames_published = 0

    def write(self, result: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("JPEG preview sink is closed")
        cv2: Any = importlib.import_module("cv2")
        image, states = _render_result(result, self._track_state_provider, self._track_renderer)
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if not success:
            raise RuntimeError("failed to encode preview JPEG")
        metadata = frame_result_to_record(result)
        if states is not None:
            metadata["track_count"] = len(states)
            metadata["tracks"] = [_track_state_to_preview_record(state) for state in states]
        self.hub.publish(encoded.tobytes(), metadata)
        self.frames_published += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close_hub_on_close:
            self.hub.close()


def _render_result(
    result: FrameResult,
    provider: TrackStateProvider | None,
    renderer: TrackOverlayRenderer | None,
) -> tuple[Any, tuple[TrackState, ...] | None]:
    if provider is None:
        return render_annotated_frame(result), None
    if renderer is None:
        raise RuntimeError("track renderer is required when a track state provider is configured")
    states = tuple(provider())
    return renderer.render(result, states), states


def _track_state_to_preview_record(state: TrackState) -> dict[str, object]:
    return {
        "track_id": state.track_id,
        "frame_id": state.frame_id,
        "bbox_xyxy": [round(value, 6) for value in state.bbox_xyxy],
        "class_id": state.class_id,
        "confidence": round(state.confidence, 6),
        "observation": state.observation.value,
        "lifecycle": state.lifecycle.value,
        "age_frames": state.age_frames,
        "time_since_update": state.time_since_update,
        "lost_reason": None if state.lost_reason is None else state.lost_reason.value,
    }


def _bbox_center(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width - 1, round((x1 + x2) / 2.0))),
        max(0, min(height - 1, round((y1 + y2) / 2.0))),
    )


def _clamp_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width - 1, round(x2))),
        max(0, min(height - 1, round(y2))),
    )


def _draw_label(
    cv2: Any,
    image: Any,
    label: str,
    *,
    left: int,
    top: int,
    width: int,
    color: tuple[int, int, int],
    font_scale: float,
    line_width: int,
) -> None:
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


def _draw_dashed_rectangle(
    cv2: Any,
    image: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    _draw_dashed_line(cv2, image, (left, top), (right, top), color, thickness)
    _draw_dashed_line(cv2, image, (right, top), (right, bottom), color, thickness)
    _draw_dashed_line(cv2, image, (right, bottom), (left, bottom), color, thickness)
    _draw_dashed_line(cv2, image, (left, bottom), (left, top), color, thickness)


def _draw_dashed_line(
    cv2: Any,
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_length: int = 8,
) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    distance = hypot(delta_x, delta_y)
    if distance == 0:
        return
    step = dash_length * 2
    for offset in range(0, max(1, round(distance)), step):
        segment_end = min(float(offset + dash_length), distance)
        first = (
            round(start[0] + delta_x * offset / distance),
            round(start[1] + delta_y * offset / distance),
        )
        second = (
            round(start[0] + delta_x * segment_end / distance),
            round(start[1] + delta_y * segment_end / distance),
        )
        cv2.line(image, first, second, color, thickness, cv2.LINE_AA)
