"""Strict detector-ablation plans that can be audited before allocating GPUs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DetectorExperimentRun:
    """One fully expanded, reproducible detector training run."""

    suite_id: str
    variant_id: str
    run_id: str
    model: Path
    data: Path
    initialization_weights: Path | None
    project: Path
    seed: int
    epochs: int
    image_size: int
    batch_size: int
    workers: int
    nwd_weight: float
    nwd_constant: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("suite_id", self.suite_id),
            ("variant_id", self.variant_id),
            ("run_id", self.run_id),
        ):
            if not value or not value.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"{field_name} must contain only letters, numbers, '_' or '-'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if min(self.epochs, self.image_size, self.batch_size) <= 0 or self.workers < 0:
            raise ValueError("epochs/image_size/batch_size must be positive and workers non-negative")
        if not isfinite(self.nwd_weight) or not 0.0 <= self.nwd_weight <= 1.0:
            raise ValueError("nwd_weight must be within [0, 1]")
        if not isfinite(self.nwd_constant) or self.nwd_constant <= 0:
            raise ValueError("nwd_constant must be positive")

    def to_record(self, *, root: Path) -> dict[str, object]:
        record = asdict(self)
        for key in ("model", "data", "initialization_weights", "project"):
            value = record[key]
            record[key] = None if value is None else _portable_path(Path(value), root)
        return record


@dataclass(frozen=True, slots=True)
class DetectorExperimentSuite:
    """Validated experiment suite plus its expanded run matrix."""

    schema_version: int
    suite_id: str
    description: str
    source_path: Path
    source_sha256: str
    runs: tuple[DetectorExperimentRun, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only detector experiment schema_version 1 is supported")
        if not self.suite_id or not self.description.strip():
            raise ValueError("suite_id and description must not be empty")
        if not self.runs:
            raise ValueError("experiment suite must expand to at least one run")
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("expanded experiment run IDs must be unique")

    def to_record(self, *, root: Path) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "description": self.description,
            "source_path": _portable_path(self.source_path, root),
            "source_sha256": self.source_sha256,
            "run_count": len(self.runs),
            "runs": [run.to_record(root=root) for run in self.runs],
        }


def load_detector_experiment_suite(path: Path, *, repo_root: Path | None = None) -> DetectorExperimentSuite:
    """Load and strictly expand a JSON detector-ablation suite."""

    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    root = (repo_root or source_path.parents[2]).resolve()
    try:
        payload: object = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON: {source_path}:{exc.lineno}:{exc.colno}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration root must be an object")
    _require_exact_keys(payload, {"schema_version", "suite_id", "description", "defaults", "variants", "seeds"})
    schema_version = _required_int(payload, "schema_version")
    suite_id = _required_string(payload, "suite_id")
    description = _required_string(payload, "description")
    defaults = _required_mapping(payload, "defaults")
    variants = _required_list(payload, "variants")
    seeds = _integer_list(payload, "seeds")
    if not variants:
        raise ValueError("variants must not be empty")
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")
    _require_exact_keys(
        defaults,
        {
            "model",
            "data",
            "initialization_weights",
            "project",
            "epochs",
            "image_size",
            "batch_size",
            "workers",
            "nwd_constant",
        },
    )
    model = _repo_path(root, _required_string(defaults, "model"), must_exist=True)
    data = _repo_path(root, _required_string(defaults, "data"), must_exist=True)
    initialization_value = defaults.get("initialization_weights")
    if initialization_value is not None and not isinstance(initialization_value, str):
        raise ValueError("initialization_weights must be a string or null")
    initialization_weights = (
        None if initialization_value is None else _repo_path(root, initialization_value, must_exist=True)
    )
    project = _repo_path(root, _required_string(defaults, "project"), must_exist=False)
    epochs = _required_int(defaults, "epochs")
    image_size = _required_int(defaults, "image_size")
    batch_size = _required_int(defaults, "batch_size")
    workers = _required_int(defaults, "workers")
    nwd_constant = _required_float(defaults, "nwd_constant")

    runs: list[DetectorExperimentRun] = []
    variant_ids: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("each variant must be an object")
        _require_exact_keys(variant, {"id", "description", "nwd_weight"})
        variant_id = _required_string(variant, "id")
        if variant_id in variant_ids:
            raise ValueError(f"duplicate variant id: {variant_id}")
        variant_ids.add(variant_id)
        _required_string(variant, "description")
        nwd_weight = _required_float(variant, "nwd_weight")
        for seed in seeds:
            runs.append(
                DetectorExperimentRun(
                    suite_id=suite_id,
                    variant_id=variant_id,
                    run_id=f"{variant_id}-seed-{seed}",
                    model=model,
                    data=data,
                    initialization_weights=initialization_weights,
                    project=project,
                    seed=seed,
                    epochs=epochs,
                    image_size=image_size,
                    batch_size=batch_size,
                    workers=workers,
                    nwd_weight=nwd_weight,
                    nwd_constant=nwd_constant,
                )
            )
    return DetectorExperimentSuite(
        schema_version=schema_version,
        suite_id=suite_id,
        description=description,
        source_path=source_path,
        source_sha256=sha256(source_path.read_bytes()).hexdigest(),
        runs=tuple(runs),
    )


def _required_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _integer_list(payload: dict[str, object], key: str) -> list[int]:
    values = _required_list(payload, key)
    result: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} must contain only integers")
        result.append(value)
    return result


def _require_exact_keys(payload: dict[str, object], expected: set[str]) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"configuration keys mismatch; missing={missing}, unexpected={unexpected}")


def _repo_path(root: Path, value: str, *, must_exist: bool) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"experiment path escapes repository: {value}") from exc
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
