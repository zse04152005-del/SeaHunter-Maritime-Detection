"""Quality-gated appearance observations and lightweight baseline encoding."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Protocol

from seahunter.schemas import Detection, FramePacket


class AppearanceEncoder(Protocol):
    """Pluggable crop encoder used by the BoT-SORT appearance branch."""

    @property
    def encoder_id(self) -> str:
        """Return a stable encoder/configuration identifier."""

    def encode(
        self,
        frame: FramePacket,
        detection: Detection,
        crop: Any,
    ) -> Sequence[float]:
        """Return one finite embedding for an already quality-approved crop."""


@dataclass(frozen=True, slots=True)
class AppearanceQualityConfig:
    """Image-space gates that prevent unreliable crops from entering ReID."""

    minimum_short_side_px: float = 24.0
    minimum_area_px: float = 768.0
    minimum_visible_fraction: float = 0.9
    maximum_overlap_fraction: float = 0.6
    minimum_brightness: float = 20.0
    maximum_brightness: float = 235.0
    minimum_sharpness: float = 10.0

    def __post_init__(self) -> None:
        if self.minimum_short_side_px <= 0 or self.minimum_area_px <= 0:
            raise ValueError("minimum appearance size and area must be positive")
        if not 0.0 <= self.minimum_visible_fraction <= 1.0:
            raise ValueError("minimum_visible_fraction must be within [0, 1]")
        if not 0.0 <= self.maximum_overlap_fraction <= 1.0:
            raise ValueError("maximum_overlap_fraction must be within [0, 1]")
        if not 0.0 <= self.minimum_brightness <= self.maximum_brightness <= 255.0:
            raise ValueError("brightness bounds must satisfy 0 <= minimum <= maximum <= 255")
        if self.minimum_sharpness < 0:
            raise ValueError("minimum_sharpness must be non-negative")


@dataclass(frozen=True, slots=True)
class AppearanceObservation:
    """One auditable quality decision and optional normalized embedding."""

    eligible: bool
    quality: float
    embedding: tuple[float, ...] | None
    bypass_reason: str | None
    short_side_px: float
    visible_fraction: float
    overlap_fraction: float
    brightness: float | None
    sharpness: float | None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("appearance quality must be within [0, 1]")
        if self.eligible:
            if self.embedding is None or not self.embedding:
                raise ValueError("eligible appearance observations require an embedding")
            if self.bypass_reason is not None:
                raise ValueError("eligible appearance observations cannot have a bypass reason")
        elif self.bypass_reason is None or not self.bypass_reason.strip():
            raise ValueError("ineligible appearance observations require a bypass reason")
        if self.embedding is not None and not all(isfinite(value) for value in self.embedding):
            raise ValueError("appearance embedding must contain finite values")


class HistogramAppearanceEncoder:
    """Edge-friendly color histogram baseline; replaceable by an ONNX ReID model."""

    def __init__(self, bins_per_channel: int = 8) -> None:
        if bins_per_channel < 2:
            raise ValueError("bins_per_channel must be at least 2")
        self.bins_per_channel = bins_per_channel
        self._np: Any = importlib.import_module("numpy")
        self._encoder_id = f"bgr-histogram:v1:bins={bins_per_channel}"

    @property
    def encoder_id(self) -> str:
        return self._encoder_id

    def encode(self, frame: FramePacket, detection: Detection, crop: Any) -> Sequence[float]:
        del frame, detection
        array = self._np.asarray(crop)
        channels = 1 if array.ndim == 2 else min(3, int(array.shape[2]))
        features: list[float] = []
        for channel in range(channels):
            values = array if array.ndim == 2 else array[:, :, channel]
            histogram, _ = self._np.histogram(values, bins=self.bins_per_channel, range=(0.0, 256.0))
            features.extend(float(value) for value in histogram)
        return _normalize_embedding(features)


class AppearanceQualityGate:
    """Assess crop reliability before invoking an appearance encoder."""

    def __init__(
        self,
        config: AppearanceQualityConfig | None = None,
        *,
        encoder: AppearanceEncoder,
    ) -> None:
        self.config = config or AppearanceQualityConfig()
        self.encoder = encoder
        self._np: Any = importlib.import_module("numpy")

    def observe(
        self,
        frame: FramePacket,
        detection: Detection,
        all_detections: Sequence[Detection],
    ) -> AppearanceObservation:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        short_side = min(width, height)
        visible_width = max(0.0, min(float(frame.width), x2) - max(0.0, x1))
        visible_height = max(0.0, min(float(frame.height), y2) - max(0.0, y1))
        visible_fraction = 0.0 if area <= 0.0 else visible_width * visible_height / area
        overlap_fraction = _maximum_overlap_fraction(detection, all_detections)

        if short_side < self.config.minimum_short_side_px or area < self.config.minimum_area_px:
            return self._rejected("target_too_small", short_side, visible_fraction, overlap_fraction)
        if visible_fraction < self.config.minimum_visible_fraction:
            return self._rejected("insufficient_visible_fraction", short_side, visible_fraction, overlap_fraction)
        if overlap_fraction > self.config.maximum_overlap_fraction:
            return self._rejected("occluded_overlap", short_side, visible_fraction, overlap_fraction)
        crop = self._crop(frame, detection)
        if crop is None:
            return self._rejected("invalid_or_missing_crop", short_side, visible_fraction, overlap_fraction)

        brightness, sharpness = self._image_quality(crop)
        if brightness < self.config.minimum_brightness:
            return self._rejected(
                "brightness_too_low", short_side, visible_fraction, overlap_fraction, brightness, sharpness
            )
        if brightness > self.config.maximum_brightness:
            return self._rejected(
                "brightness_too_high", short_side, visible_fraction, overlap_fraction, brightness, sharpness
            )
        if sharpness < self.config.minimum_sharpness:
            return self._rejected(
                "crop_too_blurry", short_side, visible_fraction, overlap_fraction, brightness, sharpness
            )

        embedding = _normalize_embedding(self.encoder.encode(frame, detection, crop))
        size_quality = min(1.0, short_side / (2.0 * self.config.minimum_short_side_px))
        visibility_quality = min(1.0, visible_fraction)
        overlap_quality = max(0.0, 1.0 - overlap_fraction)
        sharpness_quality = (
            1.0 if self.config.minimum_sharpness == 0 else min(1.0, sharpness / (2.0 * self.config.minimum_sharpness))
        )
        quality = max(0.0, min(1.0, size_quality * visibility_quality * overlap_quality * sharpness_quality))
        return AppearanceObservation(
            eligible=True,
            quality=quality,
            embedding=embedding,
            bypass_reason=None,
            short_side_px=short_side,
            visible_fraction=visible_fraction,
            overlap_fraction=overlap_fraction,
            brightness=brightness,
            sharpness=sharpness,
        )

    def _crop(self, frame: FramePacket, detection: Detection) -> Any | None:
        if frame.payload is None:
            return None
        array = self._np.asarray(frame.payload)
        if array.ndim not in (2, 3) or array.shape[0] != frame.height or array.shape[1] != frame.width:
            return None
        if not bool(self._np.isfinite(array).all()):
            return None
        x1, y1, x2, y2 = detection.bbox_xyxy
        left = max(0, min(frame.width, int(x1)))
        top = max(0, min(frame.height, int(y1)))
        right = max(0, min(frame.width, int(x2 + 0.999999)))
        bottom = max(0, min(frame.height, int(y2 + 0.999999)))
        if right <= left or bottom <= top:
            return None
        crop = array[top:bottom, left:right]
        return crop if crop.size else None

    def _image_quality(self, crop: Any) -> tuple[float, float]:
        array = self._np.asarray(crop, dtype=float)
        gray = array if array.ndim == 2 else array[:, :, :3].mean(axis=2)
        brightness = float(gray.mean())
        gradient_x = self._np.diff(gray, axis=1)
        gradient_y = self._np.diff(gray, axis=0)
        sharpness = float(gradient_x.var() + gradient_y.var())
        return brightness, sharpness

    @staticmethod
    def _rejected(
        reason: str,
        short_side: float,
        visible_fraction: float,
        overlap_fraction: float,
        brightness: float | None = None,
        sharpness: float | None = None,
    ) -> AppearanceObservation:
        return AppearanceObservation(
            eligible=False,
            quality=0.0,
            embedding=None,
            bypass_reason=reason,
            short_side_px=short_side,
            visible_fraction=visible_fraction,
            overlap_fraction=overlap_fraction,
            brightness=brightness,
            sharpness=sharpness,
        )


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        return -1.0
    value = sum(left * right for left, right in zip(first, second, strict=True))
    return max(-1.0, min(1.0, value)) if isfinite(value) else -1.0


def aggregate_template(
    current: Sequence[float] | None,
    observation: Sequence[float],
    *,
    quality: float,
    update_rate: float,
) -> tuple[float, ...]:
    normalized = _normalize_embedding(observation)
    if current is None:
        return normalized
    if len(current) != len(normalized):
        raise ValueError("appearance template dimensions must remain stable")
    alpha = max(0.0, min(1.0, update_rate * quality))
    blended = tuple((1.0 - alpha) * old + alpha * new for old, new in zip(current, normalized, strict=True))
    return _normalize_embedding(blended)


def _normalize_embedding(values: Sequence[float]) -> tuple[float, ...]:
    embedding = tuple(float(value) for value in values)
    if not embedding or not all(isfinite(value) for value in embedding):
        raise ValueError("appearance encoder must return a non-empty finite embedding")
    norm = sqrt(sum(value * value for value in embedding))
    if norm <= 1e-12:
        raise ValueError("appearance encoder returned a zero-norm embedding")
    return tuple(value / norm for value in embedding)


def _maximum_overlap_fraction(detection: Detection, all_detections: Sequence[Detection]) -> float:
    x1, y1, x2, y2 = detection.bbox_xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return 1.0
    maximum = 0.0
    for other in all_detections:
        if other is detection:
            continue
        ox1, oy1, ox2, oy2 = other.bbox_xyxy
        intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        maximum = max(maximum, intersection / area)
    return maximum
