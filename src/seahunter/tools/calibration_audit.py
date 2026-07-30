"""Validate one camera calibration manifest without opening a camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seahunter.geometry import calibration_quality, image_pixel_to_camera_ray, load_camera_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--maximum-rms-px", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.minimum_quality <= 1.0:
        raise SystemExit("--minimum-quality must be within [0, 1]")
    calibration = load_camera_calibration(args.calibration)
    quality = calibration_quality(calibration, maximum_rms_px=args.maximum_rms_px)
    intrinsics = calibration.intrinsics
    pixels = {
        "center": (intrinsics.cx_px, intrinsics.cy_px),
        "top_left": (0.0, 0.0),
        "top_right": (float(intrinsics.width), 0.0),
        "bottom_left": (0.0, float(intrinsics.height)),
        "bottom_right": (float(intrinsics.width), float(intrinsics.height)),
    }
    report = {
        "schema_version": 1,
        "calibration_id": calibration.calibration_id,
        "camera_serial": calibration.camera_serial,
        "lens_id": calibration.lens_id,
        "calibrated_at": calibration.calibrated_at.isoformat(),
        "calibration_rms_px": intrinsics.calibration_rms_px,
        "quality": quality,
        "accepted": quality >= args.minimum_quality,
        "minimum_quality": args.minimum_quality,
        "maximum_rms_px": args.maximum_rms_px,
        "reference_rays_camera": {
            name: [round(value, 9) for value in image_pixel_to_camera_ray(pixel, intrinsics)]
            for name, pixel in pixels.items()
        },
    }
    output = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8", newline="\n")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
