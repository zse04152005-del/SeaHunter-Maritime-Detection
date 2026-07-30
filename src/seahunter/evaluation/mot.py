"""MOTChallenge validation, TrackEval orchestration, and error-frame indexing."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

MOTFileKind = Literal["ground_truth", "prediction"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class MOTRow:
    """One normalized MOTChallenge bounding-box record."""

    frame_id: int
    track_id: int
    bbox_xywh: tuple[float, float, float, float]
    confidence: float = 1.0
    mark: int = 1
    class_id: int = 1
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_id <= 0:
            raise ValueError("MOT frame_id must be positive and one-based")
        if self.track_id <= 0:
            raise ValueError("MOT track_id must be positive")
        x, y, width, height = self.bbox_xywh
        if not all(isfinite(value) for value in (x, y, width, height, self.confidence, self.visibility)):
            raise ValueError("MOT numeric values must be finite")
        if width <= 0 or height <= 0:
            raise ValueError("MOT bounding-box width and height must be positive")
        if self.mark not in (0, 1):
            raise ValueError("MOT ground-truth mark must be 0 or 1")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("MOT visibility must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class MOTSequence:
    """Validated rows from one ground-truth or prediction sequence."""

    path: Path
    kind: MOTFileKind
    rows: tuple[MOTRow, ...]

    @property
    def max_frame_id(self) -> int:
        return max((row.frame_id for row in self.rows), default=0)

    def rows_by_frame(self, *, marked_only: bool = False) -> dict[int, tuple[MOTRow, ...]]:
        grouped: dict[int, list[MOTRow]] = defaultdict(list)
        for row in self.rows:
            if marked_only and row.mark == 0:
                continue
            grouped[row.frame_id].append(row)
        return {frame_id: tuple(sorted(rows, key=lambda item: item.track_id)) for frame_id, rows in grouped.items()}


@dataclass(frozen=True, slots=True)
class MOTEvaluationConfig:
    """Reproducible TrackEval and diagnostic matching configuration."""

    sequence_name: str
    tracker_name: str = "seahunter"
    benchmark_name: str = "SEAHUNTER"
    iou_threshold: float = 0.5
    sequence_length: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("sequence_name", self.sequence_name),
            ("tracker_name", self.tracker_name),
            ("benchmark_name", self.benchmark_name),
        ):
            if not _SAFE_NAME.fullmatch(value):
                raise ValueError(f"{field_name} must contain only letters, numbers, '.', '_', or '-'")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be within (0, 1]")
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive when provided")


def load_mot_file(path: Path, *, kind: MOTFileKind) -> MOTSequence:
    """Load and strictly validate a MOTChallenge text file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[MOTRow] = []
    seen: set[tuple[int, int]] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [column for column in re.split(r"[\s,]+", line) if column]
        if len(columns) < 6:
            raise ValueError(f"{path}:{line_number}: expected at least 6 MOT columns")
        try:
            values = [float(column) for column in columns]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: MOT columns must be numeric") from exc
        if not all(isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: MOT columns must be finite")

        frame_id = _integer_column(values[0], path, line_number, "frame_id")
        track_id = _integer_column(values[1], path, line_number, "track_id")
        key = (frame_id, track_id)
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate track_id {track_id} in frame {frame_id}")
        seen.add(key)

        if kind == "ground_truth":
            mark = _integer_column(values[6], path, line_number, "mark") if len(values) >= 7 else 1
            class_id = _integer_column(values[7], path, line_number, "class_id") if len(values) >= 8 else 1
            visibility = values[8] if len(values) >= 9 else 1.0
            confidence = 1.0
        else:
            confidence = values[6] if len(values) >= 7 else 1.0
            class_id = _integer_column(values[7], path, line_number, "class_id") if len(values) >= 8 else 1
            mark = 1
            visibility = 1.0
        try:
            row = MOTRow(
                frame_id=frame_id,
                track_id=track_id,
                bbox_xywh=(values[2], values[3], values[4], values[5]),
                confidence=confidence,
                mark=mark,
                class_id=class_id,
                visibility=visibility,
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append(row)

    return MOTSequence(path=path, kind=kind, rows=tuple(sorted(rows, key=lambda row: (row.frame_id, row.track_id))))


def build_mot_error_index(
    ground_truth: MOTSequence,
    predictions: MOTSequence,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Build an auditable frame index for misses, false positives, switches, and fragments."""

    _validate_sequence_kinds(ground_truth, predictions)
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within (0, 1]")
    gt_by_frame = ground_truth.rows_by_frame(marked_only=True)
    pred_by_frame = predictions.rows_by_frame()
    frame_ids = sorted(set(gt_by_frame) | set(pred_by_frame))
    previous_prediction: dict[int, int] = {}
    ever_matched: set[int] = set()
    unmatched_after_match: set[int] = set()
    error_frames: list[dict[str, object]] = []
    total_matches = 0
    total_false_negatives = 0
    total_false_positives = 0
    total_id_switches = 0
    total_fragmentations = 0

    for frame_id in frame_ids:
        gt_rows = gt_by_frame.get(frame_id, ())
        pred_rows = pred_by_frame.get(frame_id, ())
        matches, unmatched_gt, unmatched_predictions = _associate_rows(gt_rows, pred_rows, iou_threshold)
        id_switches: list[dict[str, int]] = []
        fragmentations: list[int] = []
        matched_gt_ids: set[int] = set()
        for gt_index, prediction_index, _ in matches:
            gt_id = gt_rows[gt_index].track_id
            prediction_id = pred_rows[prediction_index].track_id
            matched_gt_ids.add(gt_id)
            previous_id = previous_prediction.get(gt_id)
            if previous_id is not None and previous_id != prediction_id:
                id_switches.append(
                    {
                        "ground_truth_id": gt_id,
                        "previous_prediction_id": previous_id,
                        "current_prediction_id": prediction_id,
                    }
                )
            if gt_id in unmatched_after_match:
                fragmentations.append(gt_id)
                unmatched_after_match.remove(gt_id)
            previous_prediction[gt_id] = prediction_id
            ever_matched.add(gt_id)

        false_negative_ids = [gt_rows[index].track_id for index in unmatched_gt]
        false_positive_ids = [pred_rows[index].track_id for index in unmatched_predictions]
        for gt_id in false_negative_ids:
            if gt_id in ever_matched:
                unmatched_after_match.add(gt_id)

        total_matches += len(matches)
        total_false_negatives += len(false_negative_ids)
        total_false_positives += len(false_positive_ids)
        total_id_switches += len(id_switches)
        total_fragmentations += len(fragmentations)
        if false_negative_ids or false_positive_ids or id_switches or fragmentations:
            error_frames.append(
                {
                    "frame_id": frame_id,
                    "false_negative_ground_truth_ids": false_negative_ids,
                    "false_positive_prediction_ids": false_positive_ids,
                    "id_switches": id_switches,
                    "fragmented_ground_truth_ids": fragmentations,
                    "matched_ground_truth_ids": sorted(matched_gt_ids),
                }
            )

    return {
        "schema_version": 1,
        "iou_threshold": iou_threshold,
        "summary": {
            "ground_truth_detections": sum(len(rows) for rows in gt_by_frame.values()),
            "prediction_detections": len(predictions.rows),
            "matches": total_matches,
            "false_negatives": total_false_negatives,
            "false_positives": total_false_positives,
            "id_switches": total_id_switches,
            "fragmentations": total_fragmentations,
            "frames_with_errors": len(error_frames),
        },
        "frames": error_frames,
    }


def evaluate_mot_sequence(
    ground_truth: MOTSequence,
    predictions: MOTSequence,
    *,
    config: MOTEvaluationConfig,
    artifacts_dir: Path,
) -> dict[str, object]:
    """Run standard TrackEval HOTA/CLEAR/Identity metrics and return normalized JSON."""

    _validate_sequence_kinds(ground_truth, predictions)
    if not ground_truth.rows:
        raise ValueError("ground-truth MOT sequence must not be empty")
    sequence_length = config.sequence_length or max(ground_truth.max_frame_id, predictions.max_frame_id)
    if sequence_length < max(ground_truth.max_frame_id, predictions.max_frame_id):
        raise ValueError("sequence_length must cover all ground-truth and prediction frames")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = artifacts_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    _write_trackeval_ground_truth(normalized_dir / "ground-truth.txt", ground_truth.rows)
    _write_trackeval_predictions(normalized_dir / "predictions.txt", predictions.rows)

    trackeval = _import_trackeval()

    with TemporaryDirectory(prefix="seahunter-trackeval-") as directory:
        workspace = Path(directory)
        gt_root = workspace / "gt"
        trackers_root = workspace / "trackers"
        split_name = f"{config.benchmark_name}-train"
        gt_path = gt_root / split_name / config.sequence_name / "gt" / "gt.txt"
        prediction_path = trackers_root / split_name / config.tracker_name / "data" / f"{config.sequence_name}.txt"
        _write_trackeval_ground_truth(gt_path, ground_truth.rows)
        _write_trackeval_predictions(prediction_path, predictions.rows)

        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update(
            {
                "BREAK_ON_ERROR": True,
                "DISPLAY_LESS_PROGRESS": True,
                "LOG_ON_ERROR": str(artifacts_dir / "trackeval-errors.log"),
                "OUTPUT_DETAILED": True,
                "OUTPUT_SUMMARY": True,
                "PLOT_CURVES": False,
                "PRINT_CONFIG": False,
                "PRINT_RESULTS": False,
                "TIME_PROGRESS": False,
                "USE_PARALLEL": False,
            }
        )
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "BENCHMARK": config.benchmark_name,
                "CLASSES_TO_EVAL": ["pedestrian"],
                "DO_PREPROC": False,
                "GT_FOLDER": str(gt_root),
                "OUTPUT_FOLDER": str(artifacts_dir / "trackeval"),
                "PRINT_CONFIG": False,
                "SEQ_INFO": {config.sequence_name: sequence_length},
                "SPLIT_TO_EVAL": "train",
                "TRACKERS_FOLDER": str(trackers_root),
                "TRACKERS_TO_EVAL": [config.tracker_name],
            }
        )
        metric_config = {
            "METRICS": ["HOTA", "CLEAR", "Identity"],
            "PRINT_CONFIG": False,
            "THRESHOLD": config.iou_threshold,
        }
        evaluator = trackeval.Evaluator(evaluator_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ]
        output_results, output_messages = evaluator.evaluate([dataset], metrics)

    dataset_results = next(iter(output_results.values()))
    message = output_messages[next(iter(output_messages))][config.tracker_name]
    if message != "Success":
        raise RuntimeError(f"TrackEval failed: {message}")
    combined = dataset_results[config.tracker_name]["COMBINED_SEQ"]["pedestrian"]
    hota = combined["HOTA"]
    clear = combined["CLEAR"]
    identity = combined["Identity"]
    counts = combined["Count"]
    diagnostics = build_mot_error_index(
        ground_truth,
        predictions,
        iou_threshold=config.iou_threshold,
    )
    return {
        "schema_version": 1,
        "evaluator": {
            "name": "TrackEval",
            "version": importlib.metadata.version("trackeval"),
        },
        "sequence": {
            "name": config.sequence_name,
            "length_frames": sequence_length,
            "ground_truth_rows": len(ground_truth.rows),
            "prediction_rows": len(predictions.rows),
        },
        "configuration": asdict(config),
        "metric_scale": "ratio_for_scores; integer_for_counts",
        "metrics": {
            "HOTA": _mean_numeric(hota["HOTA"]),
            "DetA": _mean_numeric(hota["DetA"]),
            "AssA": _mean_numeric(hota["AssA"]),
            "LocA": _mean_numeric(hota["LocA"]),
            "MOTA": _scalar_numeric(clear["MOTA"]),
            "MOTP": _scalar_numeric(clear["MOTP"]),
            "Recall": _scalar_numeric(clear["CLR_Re"]),
            "Precision": _scalar_numeric(clear["CLR_Pr"]),
            "IDF1": _scalar_numeric(identity["IDF1"]),
            "IDRecall": _scalar_numeric(identity["IDR"]),
            "IDPrecision": _scalar_numeric(identity["IDP"]),
            "IDSwitches": int(clear["IDSW"]),
            "Fragmentations": int(clear["Frag"]),
            "FalsePositives": int(clear["CLR_FP"]),
            "FalseNegatives": int(clear["CLR_FN"]),
            "GroundTruthDetections": int(counts["GT_Dets"]),
            "PredictionDetections": int(counts["Dets"]),
        },
        "diagnostics": diagnostics["summary"],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic human-readable JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _integer_column(value: float, path: Path, line_number: int, field_name: str) -> int:
    if not value.is_integer():
        raise ValueError(f"{path}:{line_number}: {field_name} must be an integer")
    return int(value)


def _validate_sequence_kinds(ground_truth: MOTSequence, predictions: MOTSequence) -> None:
    if ground_truth.kind != "ground_truth":
        raise ValueError("ground_truth sequence must be loaded with kind='ground_truth'")
    if predictions.kind != "prediction":
        raise ValueError("predictions sequence must be loaded with kind='prediction'")


def _import_trackeval() -> Any:
    """Import the pinned TrackEval revision with its required NumPy alias bridge."""

    numpy: Any = importlib.import_module("numpy")
    # The pinned official TrackEval revision still references aliases removed in
    # NumPy 1.24. Python 3.12 cannot use the older NumPy releases that exposed
    # them, so restore only the three aliases exercised by MOTChallenge metrics.
    for alias, replacement in (("bool", bool), ("float", float), ("int", int)):
        if alias not in vars(numpy):
            setattr(numpy, alias, replacement)
    try:
        return importlib.import_module("trackeval")
    except ModuleNotFoundError as exc:
        if exc.name == "trackeval":
            raise RuntimeError("MOT evaluation requires the 'evaluation' project extra") from exc
        raise


def _write_trackeval_ground_truth(path: Path, rows: tuple[MOTRow, ...]) -> None:
    lines = [
        f"{row.frame_id},{row.track_id},{row.bbox_xywh[0]:.6f},{row.bbox_xywh[1]:.6f},"
        f"{row.bbox_xywh[2]:.6f},{row.bbox_xywh[3]:.6f},{row.mark},1,{row.visibility:.6f}"
        for row in rows
    ]
    _write_lines(path, lines)


def _write_trackeval_predictions(path: Path, rows: tuple[MOTRow, ...]) -> None:
    lines = [
        f"{row.frame_id},{row.track_id},{row.bbox_xywh[0]:.6f},{row.bbox_xywh[1]:.6f},"
        f"{row.bbox_xywh[2]:.6f},{row.bbox_xywh[3]:.6f},{row.confidence:.6f},1,1,1"
        for row in rows
    ]
    _write_lines(path, lines)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("" if not lines else "\n".join(lines) + "\n", encoding="utf-8")


def _associate_rows(
    ground_truth: tuple[MOTRow, ...],
    predictions: tuple[MOTRow, ...],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not ground_truth or not predictions:
        return [], list(range(len(ground_truth))), list(range(len(predictions)))
    np: Any = importlib.import_module("numpy")
    optimize: Any = importlib.import_module("scipy.optimize")
    scores = np.zeros((len(ground_truth), len(predictions)), dtype=float)
    for gt_index, gt_row in enumerate(ground_truth):
        for prediction_index, prediction_row in enumerate(predictions):
            scores[gt_index, prediction_index] = _bbox_iou_xywh(gt_row.bbox_xywh, prediction_row.bbox_xywh)
    row_indices, column_indices = optimize.linear_sum_assignment(1.0 - scores)
    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for gt_index, prediction_index in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
        score = float(scores[gt_index, prediction_index])
        if score < iou_threshold:
            continue
        matches.append((gt_index, prediction_index, score))
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
    return (
        matches,
        [index for index in range(len(ground_truth)) if index not in matched_gt],
        [index for index in range(len(predictions)) if index not in matched_predictions],
    )


def _bbox_iou_xywh(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_right = first[0] + first[2]
    first_bottom = first[1] + first[3]
    second_right = second[0] + second[2]
    second_bottom = second[1] + second[3]
    intersection_width = max(0.0, min(first_right, second_right) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _mean_numeric(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return round(sum(float(item) for item in value) / len(value), 6) if value else 0.0
    return _scalar_numeric(value)


def _scalar_numeric(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return round(float(value), 6)
