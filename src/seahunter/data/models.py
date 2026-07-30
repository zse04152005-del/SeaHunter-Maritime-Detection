"""Versioned maritime dataset contracts independent of annotation tooling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from pathlib import Path


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class WeatherCondition(str, Enum):
    NORMAL = "normal"
    LOW_LIGHT = "low_light"
    FOG = "fog"
    GLARE = "glare"
    HIGH_WAVE = "high_wave"


class DegradationSource(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class OcclusionLevel(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    FULL = "full"


class HardNegativeKind(str, Enum):
    WAVE = "wave"
    WAKE = "wake"
    BIRD = "bird"
    FOAM = "foam"
    REFLECTION = "reflection"


@dataclass(frozen=True, slots=True)
class ObjectAnnotation:
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    track_id: int | None = None
    occlusion: OcclusionLevel = OcclusionLevel.NONE

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(isfinite(value) for value in self.bbox_xyxy) or x2 <= x1 or y2 <= y1:
            raise ValueError("annotation bbox must be finite with positive width and height")
        if not self.class_name.strip():
            raise ValueError("annotation class_name must not be empty")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("annotation track_id must be non-negative")

    @property
    def pixel_area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    video_id: str
    voyage_id: str
    location_id: str
    captured_at: datetime
    split: DatasetSplit
    weather: WeatherCondition
    degradation_source: DegradationSource
    degradation_level: int
    width: int
    height: int
    annotations: tuple[ObjectAnnotation, ...] = ()
    hard_negatives: tuple[HardNegativeKind, ...] = ()

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.sample_id, self.video_id, self.voyage_id, self.location_id)):
            raise ValueError("dataset identity fields must not be empty")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if not 0 <= self.degradation_level <= 3:
            raise ValueError("degradation_level must be within [0, 3]")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sample dimensions must be positive")
        for annotation in self.annotations:
            x1, y1, x2, y2 = annotation.bbox_xyxy
            if x1 < 0.0 or y1 < 0.0 or x2 > self.width or y2 > self.height:
                raise ValueError("annotation bbox must remain inside the image")
        if len(set(self.hard_negatives)) != len(self.hard_negatives):
            raise ValueError("hard-negative labels must be unique per sample")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    created_at: datetime
    samples: tuple[DatasetSample, ...]
    dvc_revision: str | None = None
    lakefs_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.version.strip():
            raise ValueError("dataset_id and version must not be empty")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        if self.dvc_revision is not None and not self.dvc_revision.strip():
            raise ValueError("dvc_revision must not be empty")
        if self.lakefs_commit is not None and not self.lakefs_commit.strip():
            raise ValueError("lakefs_commit must not be empty")


def load_dataset_manifest(path: Path) -> DatasetManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest root must be an object")
    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list):
        raise ValueError("dataset manifest samples must be a list")
    samples: list[DatasetSample] = []
    for raw_sample in samples_payload:
        if not isinstance(raw_sample, dict):
            raise ValueError("each dataset sample must be an object")
        raw_annotations = raw_sample.get("annotations", [])
        raw_hard_negatives = raw_sample.get("hard_negatives", [])
        if not isinstance(raw_annotations, list) or not isinstance(raw_hard_negatives, list):
            raise ValueError("annotations and hard_negatives must be lists")
        annotations = tuple(
            ObjectAnnotation(
                bbox_xyxy=_bbox(annotation["bbox_xyxy"]),
                class_name=str(annotation["class_name"]),
                track_id=None if annotation.get("track_id") is None else int(annotation["track_id"]),
                occlusion=OcclusionLevel(str(annotation.get("occlusion", "none"))),
            )
            for annotation in raw_annotations
            if isinstance(annotation, dict)
        )
        if len(annotations) != len(raw_annotations):
            raise ValueError("each annotation must be an object")
        samples.append(
            DatasetSample(
                sample_id=str(raw_sample["sample_id"]),
                video_id=str(raw_sample["video_id"]),
                voyage_id=str(raw_sample["voyage_id"]),
                location_id=str(raw_sample["location_id"]),
                captured_at=_timestamp(raw_sample["captured_at"]),
                split=DatasetSplit(str(raw_sample["split"])),
                weather=WeatherCondition(str(raw_sample["weather"])),
                degradation_source=DegradationSource(str(raw_sample["degradation_source"])),
                degradation_level=int(raw_sample["degradation_level"]),
                width=int(raw_sample["width"]),
                height=int(raw_sample["height"]),
                annotations=annotations,
                hard_negatives=tuple(HardNegativeKind(str(value)) for value in raw_hard_negatives),
            )
        )
    return DatasetManifest(
        dataset_id=str(payload["dataset_id"]),
        version=str(payload["version"]),
        created_at=_timestamp(payload["created_at"]),
        samples=tuple(samples),
        dvc_revision=None if payload.get("dvc_revision") is None else str(payload["dvc_revision"]),
        lakefs_commit=None if payload.get("lakefs_commit") is None else str(payload["lakefs_commit"]),
    )


def _bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_xyxy must be a four-element list")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)
