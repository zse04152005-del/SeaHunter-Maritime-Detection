"""Three-seed deployability-aware innovation ablation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class AblationMetrics:
    precision: float
    recall: float
    false_alarms_per_hour: float
    parameters_millions: float
    flops_giga: float
    peak_vram_mb: float
    tensorrt_latency_ms: float

    def __post_init__(self) -> None:
        values = (
            self.precision,
            self.recall,
            self.false_alarms_per_hour,
            self.parameters_millions,
            self.flops_giga,
            self.peak_vram_mb,
            self.tensorrt_latency_ms,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("ablation metrics must be finite and non-negative")
        if self.precision > 1.0 or self.recall > 1.0:
            raise ValueError("ablation precision and recall must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class AblationRun:
    variant: str
    seed: int
    metrics: AblationMetrics
    exportable: bool

    def __post_init__(self) -> None:
        if not self.variant.strip() or self.seed < 0:
            raise ValueError("ablation run identity is invalid")


@dataclass(frozen=True, slots=True)
class MetricSummary:
    mean: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class AblationSummary:
    variant: str
    seeds: tuple[int, ...]
    metrics: dict[str, MetricSummary]
    exportable: bool


def summarize_ablations(runs: tuple[AblationRun, ...], *, minimum_seeds: int = 3) -> tuple[AblationSummary, ...]:
    if minimum_seeds <= 0:
        raise ValueError("minimum_seeds must be positive")
    variants = sorted({run.variant for run in runs})
    summaries: list[AblationSummary] = []
    for variant in variants:
        selected = tuple(run for run in runs if run.variant == variant)
        seeds = tuple(sorted(run.seed for run in selected))
        if len(set(seeds)) != len(seeds) or len(seeds) < minimum_seeds:
            raise ValueError(f"ablation variant {variant} requires {minimum_seeds} unique seeds")
        metric_names = tuple(AblationMetrics.__dataclass_fields__)
        summaries.append(
            AblationSummary(
                variant=variant,
                seeds=seeds,
                metrics={
                    name: _summary(tuple(float(getattr(run.metrics, name)) for run in selected))
                    for name in metric_names
                },
                exportable=all(run.exportable for run in selected),
            )
        )
    return tuple(summaries)


def _summary(values: tuple[float, ...]) -> MetricSummary:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return MetricSummary(mean, sqrt(variance))
