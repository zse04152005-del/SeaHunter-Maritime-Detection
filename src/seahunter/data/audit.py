"""Leakage, distribution, and hard-negative audits for maritime manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import DatasetManifest, DatasetSample, DatasetSplit, WeatherCondition


@dataclass(frozen=True, slots=True)
class DatasetLeak:
    group_kind: str
    group_id: str
    splits: tuple[DatasetSplit, ...]


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    sample_count: int
    annotation_count: int
    split_samples: dict[str, int]
    class_counts: dict[str, int]
    pixel_size_buckets: dict[str, int]
    hard_negative_counts: dict[str, int]
    weather_split_counts: dict[str, dict[str, int]]
    degradation_counts: dict[str, int]
    leakage: tuple[DatasetLeak, ...]
    missing_test_weather: tuple[WeatherCondition, ...]
    long_tail_ratio: float | None
    reproducible_data_version: bool

    @property
    def passes_blocking_gates(self) -> bool:
        return not self.leakage and not self.missing_test_weather and self.reproducible_data_version

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "annotation_count": self.annotation_count,
            "split_samples": self.split_samples,
            "class_counts": self.class_counts,
            "pixel_size_buckets": self.pixel_size_buckets,
            "hard_negative_counts": self.hard_negative_counts,
            "weather_split_counts": self.weather_split_counts,
            "degradation_counts": self.degradation_counts,
            "leakage": [
                {"group_kind": leak.group_kind, "group_id": leak.group_id, "splits": [s.value for s in leak.splits]}
                for leak in self.leakage
            ],
            "missing_test_weather": [weather.value for weather in self.missing_test_weather],
            "long_tail_ratio": self.long_tail_ratio,
            "reproducible_data_version": self.reproducible_data_version,
            "passes_blocking_gates": self.passes_blocking_gates,
        }


def audit_dataset(manifest: DatasetManifest) -> DatasetAuditReport:
    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    degradation_counts: Counter[str] = Counter()
    weather_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_splits: dict[tuple[str, str], set[DatasetSplit]] = defaultdict(set)

    for sample in manifest.samples:
        split_samples[sample.split.value] += 1
        degradation_counts[f"{sample.degradation_source.value}:{sample.degradation_level}"] += 1
        weather_split_counts[sample.weather.value][sample.split.value] += 1
        for group_kind, group_id in _blocking_groups(sample):
            group_splits[(group_kind, group_id)].add(sample.split)
        for annotation in sample.annotations:
            class_counts[annotation.class_name] += 1
            size_counts[pixel_size_bucket(annotation.pixel_area)] += 1
        for kind in sample.hard_negatives:
            hard_negative_counts[kind.value] += 1

    leakage = tuple(
        DatasetLeak(kind, group_id, tuple(sorted(splits, key=lambda value: value.value)))
        for (kind, group_id), splits in sorted(group_splits.items())
        if len(splits) > 1
    )
    test_weather = {sample.weather for sample in manifest.samples if sample.split is DatasetSplit.TEST}
    missing_test_weather = tuple(weather for weather in WeatherCondition if weather not in test_weather)
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    long_tail_ratio = None if not non_zero_counts else max(non_zero_counts) / min(non_zero_counts)
    return DatasetAuditReport(
        sample_count=len(manifest.samples),
        annotation_count=sum(class_counts.values()),
        split_samples=dict(sorted(split_samples.items())),
        class_counts=dict(sorted(class_counts.items())),
        pixel_size_buckets={name: size_counts.get(name, 0) for name in ("tiny", "small", "medium", "large")},
        hard_negative_counts=dict(sorted(hard_negative_counts.items())),
        weather_split_counts={key: dict(sorted(value.items())) for key, value in sorted(weather_split_counts.items())},
        degradation_counts=dict(sorted(degradation_counts.items())),
        leakage=leakage,
        missing_test_weather=missing_test_weather,
        long_tail_ratio=long_tail_ratio,
        reproducible_data_version=bool(manifest.dvc_revision or manifest.lakefs_commit),
    )


def assert_no_test_leakage(report: DatasetAuditReport) -> None:
    if report.leakage:
        details = ", ".join(f"{leak.group_kind}:{leak.group_id}" for leak in report.leakage)
        raise ValueError(f"dataset split leakage detected: {details}")


def pixel_size_bucket(pixel_area: float) -> str:
    if pixel_area <= 0.0:
        raise ValueError("pixel_area must be positive")
    if pixel_area < 16.0**2:
        return "tiny"
    if pixel_area < 32.0**2:
        return "small"
    if pixel_area < 96.0**2:
        return "medium"
    return "large"


def _blocking_groups(sample: DatasetSample) -> tuple[tuple[str, str], ...]:
    return (
        ("video", sample.video_id),
        ("voyage", sample.voyage_id),
        ("date_location", f"{sample.captured_at.date().isoformat()}@{sample.location_id}"),
    )
