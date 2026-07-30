"""Static integrity checks for the retained SeaHunter V1 baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_WEIGHT_SHA256 = "d8ff23bdcd3a707b174ee2ad6274518c46cc0108596bfbfd673fe0535cbeb28f"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_baseline(repo_root: Path | None = None) -> dict[str, Any]:
    """Inspect files without importing PyTorch or Ultralytics."""

    root = (repo_root or _repo_root()).resolve()
    weight = root / "weights" / "seahunter_best.pt"
    block = root / "yolo_source" / "ultralytics" / "nn" / "modules" / "block.py"
    loss = root / "yolo_source" / "ultralytics" / "utils" / "loss.py"
    tasks = root / "yolo_source" / "ultralytics" / "nn" / "tasks.py"
    data_init = root / "yolo_source" / "ultralytics" / "data" / "__init__.py"
    version_file = root / "yolo_source" / "ultralytics" / "__init__.py"

    required = [weight, block, loss, tasks, data_init, version_file]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]

    weight_sha256 = _sha256(weight) if weight.is_file() else None
    block_text = block.read_text(encoding="utf-8") if block.is_file() else ""
    loss_text = loss.read_text(encoding="utf-8") if loss.is_file() else ""
    tasks_text = tasks.read_text(encoding="utf-8") if tasks.is_file() else ""
    version_text = version_file.read_text(encoding="utf-8") if version_file.is_file() else ""

    checks = {
        "weight_sha256": weight_sha256 == EXPECTED_WEIGHT_SHA256,
        "version_8_3_234": '__version__ = "8.3.234"' in version_text,
        "spdconv_present": "class SPDConv" in block_text,
        "ema_present": "class EMA" in block_text,
        "multiseam_present": "class MultiSEAM" in block_text,
        "custom_modules_registered": all(name in tasks_text for name in ("SPDConv", "EMA", "MultiSEAM")),
        "nwd_defined_and_called": loss_text.count("wasserstein_loss(") >= 2,
        "data_package_present": data_init.is_file(),
    }

    return {
        "root": str(root),
        "missing": missing,
        "weight": {
            "path": str(weight.relative_to(root)),
            "sha256": weight_sha256,
            "expected_sha256": EXPECTED_WEIGHT_SHA256,
            "size_bytes": weight.stat().st_size if weight.is_file() else None,
        },
        "checks": checks,
        "ok": not missing and all(checks.values()),
    }


def main() -> int:
    result = audit_baseline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
