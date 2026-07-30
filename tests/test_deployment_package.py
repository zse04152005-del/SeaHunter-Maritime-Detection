from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from seahunter.deployment import HMACSHA256Signer, InferencePrecision, build_model_package, verify_model_package
from seahunter.schemas import ModelManifest


def model_manifest(artifact: bytes) -> ModelManifest:
    return ModelManifest(
        model_id="seahunter-detector",
        version="1.0.0-rc1",
        framework="tensorrt",
        input_size=(1024, 1024),
        class_names=("swimmer", "boat", "jetski", "life_saving_appliances", "buoy"),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        git_commit="deadbeef",
        data_version="dvc:data-v1",
    )


class DeploymentPackageTests(unittest.TestCase):
    def test_signed_package_contains_complete_reproducible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_bytes = b"fake-engine-for-contract-testing"
            engine = root / "model.engine"
            engine.write_bytes(engine_bytes)
            target = root / "model.package.zip"
            signer = HMACSHA256Signer("site-test-key", b"a" * 32)

            manifest = build_model_package(
                target,
                model=model_manifest(engine_bytes),
                created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
                inference_backend="tensorrt",
                precision=InferencePrecision.FP16,
                preprocessing="letterbox-rgb-0-1-v1",
                input_shapes=((1, 3, 1024, 1024),),
                calibration_data_version=None,
                evaluation_metrics=(("map50", 0.5), ("false_alarms_per_hour", 1.0)),
                artifacts=(("model.engine", engine),),
                signer=signer,
            )
            verified = verify_model_package(target, signer)

            self.assertTrue(verified.valid)
            self.assertEqual(manifest.model.git_commit, "deadbeef")
            self.assertEqual(verified.manifest["precision"], "fp16")  # type: ignore[index]
            self.assertEqual(verified.manifest["model"]["data_version"], "dvc:data-v1")  # type: ignore[index]

    def test_wrong_key_and_artifact_tampering_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"original")
            package = root / "model.zip"
            signer = HMACSHA256Signer("key-a", b"a" * 32)
            build_model_package(
                package,
                model=model_manifest(b"original"),
                created_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
                inference_backend="onnxruntime",
                precision=InferencePrecision.FP32,
                preprocessing="rgb",
                input_shapes=((1, 3, 1024, 1024),),
                calibration_data_version=None,
                evaluation_metrics=(),
                artifacts=(("model.onnx", artifact),),
                signer=signer,
            )
            wrong = verify_model_package(package, HMACSHA256Signer("key-b", b"b" * 32))
            self.assertFalse(wrong.valid)
            self.assertIn("signature_identity_mismatch", wrong.errors)

            with zipfile.ZipFile(package, "a") as archive:
                archive.writestr("artifacts/model.onnx", b"tampered")
            tampered = verify_model_package(package, signer)
            self.assertFalse(tampered.valid)
            self.assertIn("artifact_hash_mismatch:model.onnx", tampered.errors)


if __name__ == "__main__":
    unittest.main()
