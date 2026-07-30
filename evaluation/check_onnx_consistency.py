#!/usr/bin/env python3
"""Compare raw PyTorch and ONNX detector outputs for a deterministic tensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seahunter.runtime import activate_legacy_ultralytics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/seahunter_best.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "weights/seahunter_best.onnx")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--max-abs", type=float, default=1e-3)
    parser.add_argument("--mean-abs", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    ultralytics = activate_legacy_ultralytics(ROOT)
    wrapper = ultralytics.YOLO(str(args.weights))
    model = wrapper.model.eval()

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    tensor = torch.rand((1, 3, args.imgsz, args.imgsz), generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = model(tensor)
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    if not isinstance(torch_output, torch.Tensor):
        raise TypeError(f"unexpected PyTorch output type: {type(torch_output)!r}")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: tensor.numpy()})[0]
    torch_array = torch_output.detach().cpu().numpy()

    same_shape = torch_array.shape == onnx_output.shape
    if same_shape:
        absolute_error = np.abs(torch_array - onnx_output)
        max_abs = float(absolute_error.max())
        mean_abs = float(absolute_error.mean())
    else:
        max_abs = float("inf")
        mean_abs = float("inf")

    passed = same_shape and max_abs <= args.max_abs and mean_abs <= args.mean_abs
    report = {
        "weights": str(args.weights),
        "onnx": str(args.onnx),
        "input_shape": list(tensor.shape),
        "torch_output_shape": list(torch_array.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_abs_threshold": args.max_abs,
        "mean_abs_threshold": args.mean_abs,
        "passed": passed,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
