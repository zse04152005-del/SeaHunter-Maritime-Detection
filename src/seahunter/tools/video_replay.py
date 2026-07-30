"""Command-line entry point for file or live-stream detection replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seahunter.runtime import RuntimeResourceMonitor, UltralyticsDetector
from seahunter.schemas import SourceKind
from seahunter.services import (
    AnnotatedVideoSink,
    FrameResultSink,
    JpegPreviewSink,
    ParquetResultSink,
    PreviewHub,
    PreviewServer,
    TrackingSink,
    TrackStateSink,
    create_preview_app,
    run_replay,
)
from seahunter.tracking import (
    AppearanceQualityConfig,
    BoTSORTConfig,
    BoTSORTTracker,
    ByteTrackConfig,
    ByteTracker,
    MOTChallengeWriter,
    MultiObjectTracker,
    SparseOpticalFlowConfig,
    TrackJsonlWriter,
)
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
    parser.add_argument("--parquet", type=Path, help="flattened Parquet output for batch evaluation")
    parser.add_argument("--annotated-video", type=Path, help="optional video with detection or track overlays")
    parser.add_argument("--output-fps", type=float, default=25.0, help="annotated video frame rate")
    parser.add_argument("--video-codec", help="four-character OpenCV codec, e.g. mp4v or MJPG")
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
    parser.add_argument("--preview", action="store_true", help="serve latest JPEG frames over WebSocket")
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", type=int, default=8000)
    parser.add_argument("--preview-quality", type=int, default=80)
    parser.add_argument("--tracker", choices=("none", "bytetrack", "botsort"), default="none")
    parser.add_argument("--tracks-jsonl", type=Path, help="auditable per-frame track-state JSONL")
    parser.add_argument("--mot-output", type=Path, help="MOTChallenge ten-column tracking output")
    parser.add_argument("--track-high-threshold", type=float, default=0.6)
    parser.add_argument("--track-low-threshold", type=float, default=0.1)
    parser.add_argument("--track-new-threshold", type=float, default=0.7)
    parser.add_argument("--track-first-iou", type=float, default=0.3)
    parser.add_argument("--track-second-iou", type=float, default=0.2)
    parser.add_argument("--track-max-lost", type=int, default=30)
    parser.add_argument("--track-minimum-hits", type=int, default=1)
    parser.add_argument("--track-frame-rate", type=float, default=30.0)
    parser.add_argument("--track-inferred-confidence-decay", type=float, default=0.9)
    parser.add_argument("--track-class-agnostic", action="store_true")
    parser.add_argument("--track-emit-lost", action="store_true")
    parser.add_argument("--track-motion-gating-disabled", action="store_true")
    parser.add_argument(
        "--track-motion-gate-threshold",
        type=float,
        default=13.276704,
        help="squared 4D Mahalanobis gate; default is the chi-square 99%% quantile",
    )
    parser.add_argument("--track-trail-length", type=int, default=30)
    parser.add_argument("--gmc-disabled", action="store_true", help="disable BoT-SORT visual GMC for ablation")
    parser.add_argument("--gmc-downscale", type=int, default=2)
    parser.add_argument("--gmc-minimum-inliers", type=int, default=12)
    parser.add_argument("--gmc-minimum-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--gmc-maximum-translation-ratio", type=float, default=0.35)
    parser.add_argument("--reid-enabled", action="store_true", help="enable quality-gated BoT-SORT appearance ReID")
    parser.add_argument("--reid-minimum-similarity", type=float, default=0.75)
    parser.add_argument("--reid-motion-gate-threshold", type=float, default=50.0)
    parser.add_argument("--reid-maximum-age", type=int, default=30)
    parser.add_argument("--reid-template-update-rate", type=float, default=0.25)
    parser.add_argument("--reid-minimum-short-side", type=float, default=24.0)
    parser.add_argument("--reid-minimum-area", type=float, default=768.0)
    parser.add_argument("--reid-minimum-visible-fraction", type=float, default=0.9)
    parser.add_argument("--reid-maximum-overlap-fraction", type=float, default=0.6)
    parser.add_argument("--reid-minimum-brightness", type=float, default=20.0)
    parser.add_argument("--reid-maximum-brightness", type=float, default=235.0)
    parser.add_argument("--reid-minimum-sharpness", type=float, default=10.0)
    parser.add_argument(
        "--mot-include-inferred",
        action="store_true",
        help="include motion-model predictions in MOT output; audit JSONL always preserves observation type",
    )
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
    _validate_output_paths(
        source.kind,
        source.location,
        {
            "JSONL output": args.output,
            "summary output": summary_path,
            "Parquet output": args.parquet,
            "annotated video": args.annotated_video,
            "track JSONL output": args.tracks_jsonl,
            "MOT output": args.mot_output,
        },
    )
    if args.tracker == "none" and (args.tracks_jsonl is not None or args.mot_output is not None):
        raise SystemExit("tracking outputs require --tracker bytetrack or botsort")
    if args.mot_include_inferred and args.mot_output is None:
        raise SystemExit("--mot-include-inferred requires --mot-output")
    if args.track_trail_length < 0:
        raise SystemExit("--track-trail-length must be non-negative")
    if args.reid_enabled and args.tracker != "botsort":
        raise SystemExit("--reid-enabled requires --tracker botsort")
    if args.reid_enabled and args.reid_maximum_age > args.track_max_lost:
        raise SystemExit("--reid-maximum-age cannot exceed --track-max-lost")
    monitor = RuntimeResourceMonitor()
    sinks: list[FrameResultSink] = []
    if args.parquet is not None:
        sinks.append(ParquetResultSink(args.parquet))
    tracking_sink: TrackingSink | None = None
    if args.tracker != "none":
        track_outputs: list[TrackStateSink] = []
        if args.tracks_jsonl is not None:
            track_outputs.append(TrackJsonlWriter(args.tracks_jsonl))
        if args.mot_output is not None:
            track_outputs.append(
                MOTChallengeWriter(
                    args.mot_output,
                    include_inferred=args.mot_include_inferred,
                )
            )
        tracking_sink = TrackingSink(
            _build_tracker(args),
            sinks=track_outputs,
        )
        sinks.append(tracking_sink)

    track_state_provider = None if tracking_sink is None else tracking_sink.snapshot
    if args.annotated_video is not None:
        sinks.append(
            AnnotatedVideoSink(
                args.annotated_video,
                fps=args.output_fps,
                codec=args.video_codec,
                track_state_provider=track_state_provider,
                track_trail_length=args.track_trail_length,
            )
        )

    server: PreviewServer | None = None
    if args.preview:
        hub = PreviewHub()
        sinks.append(
            JpegPreviewSink(
                hub,
                quality=args.preview_quality,
                track_state_provider=track_state_provider,
                track_trail_length=args.track_trail_length,
            )
        )
        app = create_preview_app(hub, metrics_provider=lambda: monitor.summary().to_dict())
        server = PreviewServer(app, host=args.preview_host, port=args.preview_port)
        server.start()
        print(f"SeaHunter preview: http://{args.preview_host}:{args.preview_port}")

    try:
        summary = run_replay(
            reader,
            detector,
            args.output,
            summary_path=summary_path,
            max_frames=args.max_frames,
            realtime=args.realtime,
            buffer_capacity=args.buffer_capacity,
            include_runtime_timings=args.include_runtime_timings,
            sinks=sinks,
            resource_monitor=monitor,
        )
    finally:
        if server is not None:
            server.stop()
    output_summary = summary.to_dict()
    if tracking_sink is not None:
        tracking_summary: dict[str, object] = {
            "tracker_id": tracking_sink.tracker.tracker_id,
            "frames_processed": tracking_sink.frames_processed,
            "states_emitted": tracking_sink.states_emitted,
        }
        if isinstance(tracking_sink.tracker, ByteTracker):
            tracking_summary["association"] = tracking_sink.tracker.association_summary()
        if isinstance(tracking_sink.tracker, BoTSORTTracker):
            tracking_summary["global_motion"] = tracking_sink.tracker.motion_summary()
            tracking_summary["appearance"] = tracking_sink.tracker.appearance_summary()
        output_summary["tracking"] = tracking_summary
    print(json.dumps(output_summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _build_tracker(args: argparse.Namespace) -> MultiObjectTracker:
    common: dict[str, Any] = {
        "high_confidence_threshold": args.track_high_threshold,
        "low_confidence_threshold": args.track_low_threshold,
        "new_track_threshold": args.track_new_threshold,
        "first_match_iou_threshold": args.track_first_iou,
        "second_match_iou_threshold": args.track_second_iou,
        "max_lost_frames": args.track_max_lost,
        "minimum_confirmed_hits": args.track_minimum_hits,
        "frame_rate": args.track_frame_rate,
        "inferred_confidence_decay": args.track_inferred_confidence_decay,
        "class_aware": not args.track_class_agnostic,
        "emit_lost_predictions": args.track_emit_lost,
        "motion_gating_enabled": not args.track_motion_gating_disabled,
        "motion_gate_threshold": args.track_motion_gate_threshold,
    }
    if args.tracker == "bytetrack":
        return ByteTracker(ByteTrackConfig(**common))
    if args.tracker == "botsort":
        motion_config = SparseOpticalFlowConfig(
            downscale_factor=args.gmc_downscale,
            minimum_inliers=args.gmc_minimum_inliers,
            minimum_inlier_ratio=args.gmc_minimum_inlier_ratio,
            maximum_translation_ratio=args.gmc_maximum_translation_ratio,
        )
        return BoTSORTTracker(
            BoTSORTConfig(
                **common,
                gmc_enabled=not args.gmc_disabled,
                global_motion=motion_config,
                reid_enabled=args.reid_enabled,
                appearance_quality=AppearanceQualityConfig(
                    minimum_short_side_px=args.reid_minimum_short_side,
                    minimum_area_px=args.reid_minimum_area,
                    minimum_visible_fraction=args.reid_minimum_visible_fraction,
                    maximum_overlap_fraction=args.reid_maximum_overlap_fraction,
                    minimum_brightness=args.reid_minimum_brightness,
                    maximum_brightness=args.reid_maximum_brightness,
                    minimum_sharpness=args.reid_minimum_sharpness,
                ),
                minimum_appearance_similarity=args.reid_minimum_similarity,
                reid_motion_gate_threshold=args.reid_motion_gate_threshold,
                maximum_reid_age_frames=args.reid_maximum_age,
                template_update_rate=args.reid_template_update_rate,
            )
        )
    raise ValueError(f"unsupported tracker: {args.tracker}")


def _validate_output_paths(
    source_kind: SourceKind,
    source_location: str,
    outputs: dict[str, Path | None],
) -> None:
    resolved: dict[Path, str] = {}
    source_path = Path(source_location).resolve() if source_kind is SourceKind.FILE else None
    for label, path in outputs.items():
        if path is None:
            continue
        target = path.resolve()
        if source_path is not None and target == source_path:
            raise SystemExit(f"{label} must not overwrite the input video")
        previous = resolved.get(target)
        if previous is not None:
            raise SystemExit(f"{label} conflicts with {previous}: {target}")
        resolved[target] = label


if __name__ == "__main__":
    raise SystemExit(main())
