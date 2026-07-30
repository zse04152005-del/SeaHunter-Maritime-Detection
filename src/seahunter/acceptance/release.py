"""V1 release manifest freeze guarded by sea-trial and external-device gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .audit import AcceptanceAudit
from .models import ExternalReleaseGates


@dataclass(frozen=True, slots=True)
class ReleaseAssets:
    version: str
    git_commit: str
    model_package_sha256: str
    tensorrt_engine_sha256: str
    edge_image_digest: str
    documentation_commit: str
    sea_trial_report_sha256: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.version, self.git_commit, self.edge_image_digest, self.documentation_commit)
        ):
            raise ValueError("release asset identity fields must not be empty")
        hashes = (self.model_package_sha256, self.tensorrt_engine_sha256, self.sea_trial_report_sha256)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("release asset hashes must be SHA-256 digests")


def freeze_release_manifest(
    output: Path,
    *,
    assets: ReleaseAssets,
    acceptance: AcceptanceAudit,
    external_gates: ExternalReleaseGates,
    frozen_at: datetime,
) -> None:
    if not acceptance.passed:
        raise RuntimeError("cannot freeze V1: sea-trial acceptance has not passed")
    if not external_gates.passed:
        raise RuntimeError("cannot freeze V1: external hardware/data gates have not passed")
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError("release frozen_at must be timezone-aware")
    payload = {
        "schema_version": 1,
        "frozen_at": frozen_at.isoformat(),
        "assets": asdict(assets),
        "acceptance": acceptance.to_dict(),
        "external_gates": asdict(external_gates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
