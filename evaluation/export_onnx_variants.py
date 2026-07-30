#!/usr/bin/env python3
"""Export and structurally audit static and dynamic-batch ONNX detector graphs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seahunter.runtime import activate_legacy_ultralytics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/seahunter_best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.opset < 17:
        raise ValueError("ONNX image size and opset are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ultralytics: Any = activate_legacy_ultralytics(ROOT)
    variants: list[dict[str, object]] = []
    for name, dynamic in (("static", False), ("dynamic-batch", True)):
        model = ultralytics.YOLO(str(args.weights))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                dynamic=dynamic,
                simplify=False,
                device="cpu",
            )
        )
        target = args.output_dir / f"seahunter-{name}.onnx"
        shutil.copy2(exported, target)
        variants.append(_audit_graph(target, dynamic=dynamic))
    report = {"schema_version": 1, "weights": str(args.weights), "imgsz": args.imgsz, "variants": variants}
    (args.output_dir / "onnx-export-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


def _audit_graph(path: Path, *, dynamic: bool) -> dict[str, object]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dimension.dim_param or int(dimension.dim_value) for dimension in dimensions]
    batch_dynamic = bool(dimensions[0].dim_param)
    if batch_dynamic != dynamic:
        raise ValueError(f"unexpected ONNX batch axis for {path}: {input_shape}")
    return {
        "name": path.name,
        "dynamic_batch": batch_dynamic,
        "input_shape": input_shape,
        "opset": max(item.version for item in model.opset_import),
        "size_bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
