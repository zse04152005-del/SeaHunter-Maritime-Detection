"""Selective high-resolution ROI scheduling with bounded resource adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

Box = tuple[float, float, float, float]


class ROISource(str, Enum):
    OBJECTNESS = "objectness"
    MOTION = "motion"
    ACTIVE_TRACK = "active_track"
    LOST_TRACK = "lost_track"


@dataclass(frozen=True, slots=True)
class ROIProposal:
    bbox_xyxy: Box
    score: float
    source: ROISource

    def __post_init__(self) -> None:
        _validate_box(self.bbox_xyxy)
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("ROI score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ResourceState:
    latency_ms: float
    target_latency_ms: float
    temperature_c: float | None
    queue_depth: int

    def __post_init__(self) -> None:
        if not isfinite(self.latency_ms) or not isfinite(self.target_latency_ms) or self.target_latency_ms <= 0.0:
            raise ValueError("resource latency values must be finite and target latency positive")
        if self.temperature_c is not None and not isfinite(self.temperature_c):
            raise ValueError("resource temperature must be finite")
        if self.queue_depth < 0:
            raise ValueError("resource queue_depth must be non-negative")


@dataclass(frozen=True, slots=True)
class SelectiveROIConfig:
    top_k: int = 4
    minimum_top_k: int = 1
    high_resolution: int = 1024
    reduced_resolution: int = 768
    global_refresh_interval: int = 30
    roi_padding_fraction: float = 0.2
    merge_iou: float = 0.5
    thermal_limit_c: float = 80.0
    queue_limit: int = 3

    def __post_init__(self) -> None:
        if self.top_k <= 0 or not 1 <= self.minimum_top_k <= self.top_k:
            raise ValueError("ROI top-K values are invalid")
        if self.high_resolution <= 0 or not 0 < self.reduced_resolution <= self.high_resolution:
            raise ValueError("ROI resolutions are invalid")
        if self.global_refresh_interval <= 0 or self.queue_limit < 0:
            raise ValueError("ROI refresh interval and queue limit are invalid")
        if not 0.0 <= self.roi_padding_fraction <= 1.0 or not 0.0 <= self.merge_iou <= 1.0:
            raise ValueError("ROI padding and merge IoU must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class PerceptionPlan:
    frame_index: int
    global_refresh: bool
    global_scan_resolution: int
    roi_resolution: int
    roi_budget: int
    rois: tuple[ROIProposal, ...]
    degradation_reasons: tuple[str, ...]


class SelectiveROIScheduler:
    def __init__(self, config: SelectiveROIConfig | None = None) -> None:
        self.config = config or SelectiveROIConfig()

    def plan(
        self,
        *,
        frame_index: int,
        frame_size: tuple[int, int],
        proposals: tuple[ROIProposal, ...],
        resources: ResourceState,
    ) -> PerceptionPlan:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        budget, resolution, reasons = self._adapt(resources)
        padded = tuple(self._pad_and_clip(proposal, width, height) for proposal in proposals)
        merged = _merge_proposals(padded, self.config.merge_iou)
        ranked = tuple(sorted(merged, key=lambda item: (-item.score, item.source.value))[:budget])
        return PerceptionPlan(
            frame_index=frame_index,
            global_refresh=frame_index % self.config.global_refresh_interval == 0,
            global_scan_resolution=self.config.reduced_resolution,
            roi_resolution=resolution,
            roi_budget=budget,
            rois=ranked,
            degradation_reasons=reasons,
        )

    def _adapt(self, resources: ResourceState) -> tuple[int, int, tuple[str, ...]]:
        pressure = resources.latency_ms / resources.target_latency_ms
        reasons: list[str] = []
        if pressure > 1.0:
            reasons.append("latency_pressure")
        if resources.queue_depth > self.config.queue_limit:
            reasons.append("queue_pressure")
        if resources.temperature_c is not None and resources.temperature_c >= self.config.thermal_limit_c:
            reasons.append("thermal_pressure")
        if not reasons:
            return self.config.top_k, self.config.high_resolution, ()
        severity = min(1.0, max(0.0, pressure - 1.0) + 0.25 * len(reasons))
        span = self.config.top_k - self.config.minimum_top_k
        budget = max(self.config.minimum_top_k, self.config.top_k - round(span * severity))
        return budget, self.config.reduced_resolution, tuple(reasons)

    def _pad_and_clip(self, proposal: ROIProposal, width: int, height: int) -> ROIProposal:
        x1, y1, x2, y2 = proposal.bbox_xyxy
        pad_x = (x2 - x1) * self.config.roi_padding_fraction
        pad_y = (y2 - y1) * self.config.roi_padding_fraction
        return ROIProposal(
            (max(0.0, x1 - pad_x), max(0.0, y1 - pad_y), min(float(width), x2 + pad_x), min(float(height), y2 + pad_y)),
            proposal.score,
            proposal.source,
        )


@dataclass(frozen=True, slots=True)
class ScaleDetection:
    bbox_xyxy: Box
    class_id: int
    confidence: float
    source: str

    def __post_init__(self) -> None:
        _validate_box(self.bbox_xyxy)
        if self.class_id < 0 or not 0.0 <= self.confidence <= 1.0 or not self.source.strip():
            raise ValueError("scale detection fields are invalid")


def fuse_scale_detections(
    detections: tuple[ScaleDetection, ...], *, iou_threshold: float = 0.6
) -> tuple[ScaleDetection, ...]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("fusion IoU threshold must be within [0, 1]")
    kept: list[ScaleDetection] = []
    for detection in sorted(detections, key=lambda item: (-item.confidence, item.class_id, item.source)):
        if any(
            detection.class_id == selected.class_id
            and _box_iou(detection.bbox_xyxy, selected.bbox_xyxy) >= iou_threshold
            for selected in kept
        ):
            continue
        kept.append(detection)
    return tuple(kept)


def _merge_proposals(proposals: tuple[ROIProposal, ...], threshold: float) -> tuple[ROIProposal, ...]:
    merged: list[ROIProposal] = []
    for proposal in sorted(proposals, key=lambda item: -item.score):
        match = next(
            (index for index, item in enumerate(merged) if _box_iou(item.bbox_xyxy, proposal.bbox_xyxy) >= threshold),
            None,
        )
        if match is None:
            merged.append(proposal)
            continue
        current = merged[match]
        merged[match] = ROIProposal(
            (
                min(current.bbox_xyxy[0], proposal.bbox_xyxy[0]),
                min(current.bbox_xyxy[1], proposal.bbox_xyxy[1]),
                max(current.bbox_xyxy[2], proposal.bbox_xyxy[2]),
                max(current.bbox_xyxy[3], proposal.bbox_xyxy[3]),
            ),
            max(current.score, proposal.score),
            current.source if current.score >= proposal.score else proposal.source,
        )
    return tuple(merged)


def _validate_box(box: Box) -> None:
    x1, y1, x2, y2 = box
    if not all(isfinite(value) for value in box) or x2 <= x1 or y2 <= y1:
        raise ValueError("box must be finite with positive width and height")


def _box_iou(first: Box, second: Box) -> float:
    intersection_w = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_h = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_w * intersection_h
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1e-12, first_area + second_area - intersection)
