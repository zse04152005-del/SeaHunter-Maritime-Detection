"""Command-line entry point for standard and auditable MOT evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seahunter.evaluation import (
    MOTEvaluationConfig,
    build_mot_error_index,
    evaluate_mot_sequence,
    load_mot_file,
)
from seahunter.evaluation.mot import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="normalized TrackEval JSON report")
    parser.add_argument("--errors-output", type=Path, help="per-frame error index JSON")
    parser.add_argument("--artifacts-dir", type=Path, help="TrackEval summaries and normalized MOT inputs")
    parser.add_argument("--sequence-name")
    parser.add_argument("--tracker-name", default="seahunter")
    parser.add_argument("--benchmark-name", default="SEAHUNTER")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ground_truth = load_mot_file(args.ground_truth, kind="ground_truth")
    predictions = load_mot_file(args.predictions, kind="prediction")
    sequence_name = args.sequence_name or args.ground_truth.stem
    errors_output = args.errors_output or args.output.with_name(f"{args.output.stem}.errors.json")
    artifacts_dir = args.artifacts_dir or args.output.with_name(f"{args.output.stem}.artifacts")
    config = MOTEvaluationConfig(
        sequence_name=sequence_name,
        tracker_name=args.tracker_name,
        benchmark_name=args.benchmark_name,
        iou_threshold=args.iou_threshold,
        sequence_length=args.sequence_length,
    )
    report = evaluate_mot_sequence(
        ground_truth,
        predictions,
        config=config,
        artifacts_dir=artifacts_dir,
    )
    errors = build_mot_error_index(ground_truth, predictions, iou_threshold=args.iou_threshold)
    write_json(args.output, report)
    write_json(errors_output, errors)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
