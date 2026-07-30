"""Reproducible DVC/lakeFS and MLflow experiment manifests."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    run_id: str
    git_commit: str
    data_version: str
    config_sha256: str
    seed: int
    model_id: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.run_id, self.git_commit, self.data_version, self.model_id)):
            raise ValueError("experiment provenance identity fields must not be empty")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a SHA-256 digest")
        if self.seed < 0:
            raise ValueError("experiment seed must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def build_experiment_provenance(
    *, run_id: str, git_commit: str, data_version: str, config_path: Path, seed: int, model_id: str
) -> ExperimentProvenance:
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return ExperimentProvenance(run_id, git_commit, data_version, digest, seed, model_id)


class MLflowExperimentRecorder:
    """Small adapter around MLflow; import remains optional for edge installs."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        if not tracking_uri.strip() or not experiment_name.strip():
            raise ValueError("MLflow tracking URI and experiment name must not be empty")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def record(
        self,
        provenance: ExperimentProvenance,
        *,
        parameters: Mapping[str, str | int | float | bool],
        metrics: Mapping[str, float],
        artifacts: tuple[Path, ...] = (),
    ) -> None:
        try:
            mlflow: Any = importlib.import_module("mlflow")
        except ModuleNotFoundError as exc:
            raise RuntimeError("MLflow recording requires the optional mlflow package") from exc
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=provenance.run_id):
            mlflow.set_tags({"git_commit": provenance.git_commit, "data_version": provenance.data_version})
            mlflow.log_params({**parameters, "seed": provenance.seed, "config_sha256": provenance.config_sha256})
            mlflow.log_metrics(dict(metrics))
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))


def write_provenance(path: Path, provenance: ExperimentProvenance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
