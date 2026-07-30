#!/usr/bin/env python3
"""Configuration-driven SeaHunter detector training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seahunter.runtime import activate_legacy_ultralytics  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "configs/models/seahunter_v1.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/data/seadronessee.example.yaml")
    parser.add_argument("--weights", type=Path, default=None, help="Optional initialization weight")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="seahunter_v1")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--nwd-weight", type=float, default=0.5)
    parser.add_argument("--nwd-constant", type=float, default=12.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    ultralytics = activate_legacy_ultralytics(ROOT)
    model = ultralytics.YOLO(str(args.model))
    if args.weights is not None:
        model.load(str(args.weights))

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=str(args.project),
        name=args.name,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.1,
        degrees=0.0,
        patience=30,
        exist_ok=False,
        amp=True,
        nwd_weight=args.nwd_weight,
        nwd_constant=args.nwd_constant,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
