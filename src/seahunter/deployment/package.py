"""Signed, self-describing model packages with artifact integrity verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from seahunter.schemas import ModelManifest

from .specs import InferencePrecision


@dataclass(frozen=True, slots=True)
class DeploymentModelManifest:
    model: ModelManifest
    created_at: datetime
    inference_backend: str
    precision: InferencePrecision
    preprocessing: str
    input_shapes: tuple[tuple[int, int, int, int], ...]
    calibration_data_version: str | None
    evaluation_metrics: tuple[tuple[str, float], ...]
    artifact_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("deployment manifest created_at must be timezone-aware")
        if not self.inference_backend.strip() or not self.preprocessing.strip() or not self.input_shapes:
            raise ValueError("deployment backend, preprocessing, and input shapes are required")
        if any(len(shape) != 4 or any(value <= 0 for value in shape) for shape in self.input_shapes):
            raise ValueError("deployment input shapes must contain four positive dimensions")
        if self.precision is InferencePrecision.INT8 and not self.calibration_data_version:
            raise ValueError("INT8 deployment manifests require a calibration data version")
        names = [name for name, _ in self.artifact_sha256]
        if not names or len(set(names)) != len(names):
            raise ValueError("deployment artifacts must be non-empty and unique")
        if any(len(digest) != 64 for _, digest in self.artifact_sha256):
            raise ValueError("deployment artifact hashes must be SHA-256 digests")

    def to_dict(self) -> dict[str, object]:
        model = asdict(self.model)
        return {
            "schema_version": 1,
            "model": model,
            "created_at": self.created_at.isoformat(),
            "inference_backend": self.inference_backend,
            "precision": self.precision.value,
            "preprocessing": self.preprocessing,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "calibration_data_version": self.calibration_data_version,
            "evaluation_metrics": dict(self.evaluation_metrics),
            "artifact_sha256": dict(self.artifact_sha256),
        }


class PackageSigner(Protocol):
    @property
    def algorithm(self) -> str: ...

    @property
    def key_id(self) -> str: ...

    def sign(self, payload: bytes) -> str: ...

    def verify(self, payload: bytes, signature: str) -> bool: ...


class HMACSHA256Signer:
    """Offline test/site signer; production may inject an HSM-backed asymmetric signer."""

    def __init__(self, key_id: str, secret: bytes) -> None:
        if not key_id.strip() or len(secret) < 32:
            raise ValueError("package signer requires a key ID and at least 32 secret bytes")
        self._key_id = key_id
        self._secret = bytes(secret)

    @property
    def algorithm(self) -> str:
        return "hmac-sha256"

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


@dataclass(frozen=True, slots=True)
class PackageVerification:
    valid: bool
    errors: tuple[str, ...]
    manifest: dict[str, object] | None


def build_model_package(
    target: Path,
    *,
    model: ModelManifest,
    created_at: datetime,
    inference_backend: str,
    precision: InferencePrecision,
    preprocessing: str,
    input_shapes: tuple[tuple[int, int, int, int], ...],
    calibration_data_version: str | None,
    evaluation_metrics: tuple[tuple[str, float], ...],
    artifacts: tuple[tuple[str, Path], ...],
    signer: PackageSigner,
) -> DeploymentModelManifest:
    if len({name for name, _ in artifacts}) != len(artifacts):
        raise ValueError("model package artifact names must be unique")
    if any(not name.strip() or "/" in name or "\\" in name or name in {".", ".."} for name, _ in artifacts):
        raise ValueError("model package artifact names must be safe flat names")
    artifact_payloads = tuple((name, path.read_bytes()) for name, path in artifacts)
    manifest = DeploymentModelManifest(
        model=model,
        created_at=created_at,
        inference_backend=inference_backend,
        precision=precision,
        preprocessing=preprocessing,
        input_shapes=input_shapes,
        calibration_data_version=calibration_data_version,
        evaluation_metrics=evaluation_metrics,
        artifact_sha256=tuple((name, hashlib.sha256(payload).hexdigest()) for name, payload in artifact_payloads),
    )
    manifest_bytes = (json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    signature = {
        "algorithm": signer.algorithm,
        "key_id": signer.key_id,
        "signature": signer.sign(manifest_bytes),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("signature.json", json.dumps(signature, sort_keys=True, separators=(",", ":")) + "\n")
        for name, payload in artifact_payloads:
            archive.writestr(f"artifacts/{name}", payload)
    temporary.replace(target)
    return manifest


def verify_model_package(path: Path, signer: PackageSigner) -> PackageVerification:
    errors: list[str] = []
    manifest: dict[str, object] | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            manifest_bytes = archive.read("manifest.json")
            raw_manifest = json.loads(manifest_bytes)
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
            else:
                errors.append("manifest_not_an_object")
            raw_signature = json.loads(archive.read("signature.json"))
            if not isinstance(raw_signature, dict):
                errors.append("signature_not_an_object")
                raw_signature = {}
            if raw_signature.get("algorithm") != signer.algorithm or raw_signature.get("key_id") != signer.key_id:
                errors.append("signature_identity_mismatch")
            elif not signer.verify(manifest_bytes, str(raw_signature.get("signature", ""))):
                errors.append("signature_invalid")
            hashes = {} if manifest is None else manifest.get("artifact_sha256", {})
            if not isinstance(hashes, dict):
                errors.append("artifact_hashes_invalid")
            else:
                for name, expected in hashes.items():
                    payload = archive.read(f"artifacts/{name}")
                    if hashlib.sha256(payload).hexdigest() != expected:
                        errors.append(f"artifact_hash_mismatch:{name}")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        errors.append(f"package_unreadable:{type(exc).__name__}")
    return PackageVerification(not errors, tuple(errors), manifest)
