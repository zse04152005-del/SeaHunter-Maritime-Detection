"""Deterministic synthetic camera-pan ablation for the BoT-SORT GMC path."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from math import hypot
from pathlib import Path
from typing import Any

from seahunter.schemas import Detection, FramePacket, TrackState
from seahunter.tracking import BoTSORTConfig, BoTSORTTracker, SparseOpticalFlowConfig

from .mot import MOTRow, MOTSequence, build_mot_error_index


def run_synthetic_gmc_ablation() -> dict[str, object]:
    """Compare identical tracker settings with visual GMC disabled and enabled."""

    np: Any = importlib.import_module("numpy")
    try:
        cv2: Any = importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError("the synthetic GMC ablation requires the 'video' project extra") from exc

    width, height = 320, 240
    frame_count = 4
    translation_per_frame = (24.0, 8.0)
    rng = np.random.default_rng(20260730)
    base_image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    frames: list[FramePacket] = []
    detections: list[Detection] = []
    for frame_id in range(frame_count):
        tx = translation_per_frame[0] * frame_id
        ty = translation_per_frame[1] * frame_id
        affine = np.asarray(((1.0, 0.0, tx), (0.0, 1.0, ty)), dtype=float)
        payload = cv2.warpAffine(
            base_image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(
            FramePacket(
                source_id="synthetic-camera-pan",
                frame_id=frame_id,
                captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc) + timedelta(seconds=frame_id / 10.0),
                width=width,
                height=height,
                payload=payload,
            )
        )
        detections.append(
            Detection(
                bbox_xyxy=(80.0 + tx, 90.0 + ty, 96.0 + tx, 106.0 + ty),
                class_id=0,
                class_name="synthetic-maritime-target",
                confidence=0.9,
                detector_id="synthetic-detector:v1",
            )
        )

    common: dict[str, Any] = {
        "first_match_iou_threshold": 0.3,
        "second_match_iou_threshold": 0.2,
        "max_lost_frames": 2,
        "frame_rate": 10.0,
    }
    baseline = BoTSORTTracker(BoTSORTConfig(**common, gmc_enabled=False))
    compensated = BoTSORTTracker(
        BoTSORTConfig(
            **common,
            global_motion=SparseOpticalFlowConfig(minimum_inliers=20),
        )
    )
    baseline_states = _run_tracker(baseline, frames, detections)
    compensated_states = _run_tracker(compensated, frames, detections)
    ground_truth = _ground_truth_sequence(detections)
    baseline_diagnostics = build_mot_error_index(ground_truth, _prediction_sequence("baseline", baseline_states))
    compensated_diagnostics = build_mot_error_index(
        ground_truth,
        _prediction_sequence("compensated", compensated_states),
    )
    baseline_summary = _summary_dict(baseline_diagnostics)
    compensated_summary = _summary_dict(compensated_diagnostics)
    applied_estimates = [state for state in compensated_states[1:] if state.global_motion_applied]
    translation_errors = [
        hypot(
            (state.global_motion_affine or (0.0,) * 6)[2] - translation_per_frame[0],
            (state.global_motion_affine or (0.0,) * 6)[5] - translation_per_frame[1],
        )
        for state in applied_estimates
    ]
    baseline_switches = _integer_metric(baseline_summary, "id_switches")
    compensated_switches = _integer_metric(compensated_summary, "id_switches")
    return {
        "schema_version": 1,
        "experiment": "deterministic_synthetic_camera_translation",
        "dataset_claim": False,
        "frame_count": frame_count,
        "translation_per_frame_xy_px": list(translation_per_frame),
        "baseline_no_gmc": {
            "tracker_id": baseline.tracker_id,
            "unique_track_ids": sorted({state.track_id for state in baseline_states}),
            "diagnostics": baseline_summary,
        },
        "botsort_with_gmc": {
            "tracker_id": compensated.tracker_id,
            "unique_track_ids": sorted({state.track_id for state in compensated_states}),
            "diagnostics": compensated_summary,
            "motion": compensated.motion_summary(),
            "maximum_translation_error_px": round(max(translation_errors), 6) if translation_errors else None,
        },
        "improvement": {
            "id_switch_reduction": baseline_switches - compensated_switches,
            "baseline_id_switches": baseline_switches,
            "compensated_id_switches": compensated_switches,
        },
    }


def _run_tracker(
    tracker: BoTSORTTracker,
    frames: list[FramePacket],
    detections: list[Detection],
) -> list[TrackState]:
    states: list[TrackState] = []
    for frame, detection in zip(frames, detections, strict=True):
        emitted = tracker.update(frame, [detection])
        observed = [state for state in emitted if state.time_since_update == 0]
        if len(observed) != 1:
            raise RuntimeError("synthetic GMC ablation expected exactly one observed state per frame")
        states.append(observed[0])
    return states


def _ground_truth_sequence(detections: list[Detection]) -> MOTSequence:
    rows = tuple(
        MOTRow(
            frame_id=index + 1,
            track_id=1,
            bbox_xywh=_xyxy_to_xywh(detection.bbox_xyxy),
        )
        for index, detection in enumerate(detections)
    )
    return MOTSequence(path=Path("synthetic-camera-pan-gt.txt"), kind="ground_truth", rows=rows)


def _prediction_sequence(name: str, states: list[TrackState]) -> MOTSequence:
    rows = tuple(
        MOTRow(
            frame_id=state.frame_id + 1,
            track_id=state.track_id,
            bbox_xywh=_xyxy_to_xywh(state.bbox_xyxy),
            confidence=state.confidence,
        )
        for state in states
    )
    return MOTSequence(path=Path(f"synthetic-camera-pan-{name}.txt"), kind="prediction", rows=rows)


def _xyxy_to_xywh(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]


def _summary_dict(index: dict[str, object]) -> dict[str, object]:
    summary = index["summary"]
    if not isinstance(summary, dict):
        raise RuntimeError("MOT diagnostic summary has an invalid type")
    return summary


def _integer_metric(summary: dict[str, object], key: str) -> int:
    value = summary[key]
    if not isinstance(value, int):
        raise RuntimeError(f"MOT diagnostic metric {key!r} has an invalid type")
    return value
