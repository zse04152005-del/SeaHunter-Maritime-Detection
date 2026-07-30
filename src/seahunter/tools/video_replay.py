"""Command-line entry point for file or live-stream detection replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seahunter.runtime import UltralyticsDetector
from seahunter.services import run_replay
from seahunter.video import OpenCVFrameReader, OpenCVReaderConfig, parse_video_source


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    """Build the replay CLI parser."""

    root = _default_repo_root()
    parser = argparse.ArgumentParser(description="Replay SeaHunter detection over a video source")
    parser.add_argument("source", help="local video, device index, RTSP, SRT, or HTTP URL")
    parser.add_argument("--output", type=Path, required=True, help="deterministic JSONL metadata output")
    parser.add_argument("--summary", type=Path, help="performance summary JSON (defaults beside output)")
    parser.add_argument("--weights", type=Path, default=root / "weights/seahunter_best.pt")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--source-id", help="stable logical camera/source identifier")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--realtime", action="store_true", help="use bounded latest-frame ingestion")
    parser.add_argument("--buffer-capacity", type=int, default=2)
    parser.add_argument("--backend", choices=("auto", "ffmpeg", "gstreamer"), default="ffmpeg")
    parser.add_argument("--open-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=2.0)
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=8,
        help="retry budget per outage; use -1 for an always-on service",
    )
    parser.add_argument("--software-decode", action="store_true", help="disable hardware acceleration preference")
    parser.add_argument(
        "--include-runtime-timings",
        action="store_true",
        help="embed non-deterministic timings in JSONL",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu or 0")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the replay CLI and print its performance summary."""

    args = build_parser().parse_args(argv)
    max_reconnect_attempts = None if args.max_reconnect_attempts == -1 else args.max_reconnect_attempts
    if max_reconnect_attempts is not None and max_reconnect_attempts < 0:
        raise SystemExit("--max-reconnect-attempts must be -1 or non-negative")

    source = parse_video_source(args.source)
    config = OpenCVReaderConfig(
        backend=args.backend,
        open_timeout_seconds=args.open_timeout,
        read_timeout_seconds=args.read_timeout,
        prefer_hardware_decode=not args.software_decode,
        max_reconnect_attempts=max_reconnect_attempts,
    )
    reader = OpenCVFrameReader(source, source_id=args.source_id, config=config)
    detector = UltralyticsDetector(
        args.weights,
        repo_root=args.repo_root,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = run_replay(
        reader,
        detector,
        args.output,
        summary_path=summary_path,
        max_frames=args.max_frames,
        realtime=args.realtime,
        buffer_capacity=args.buffer_capacity,
        include_runtime_timings=args.include_runtime_timings,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
