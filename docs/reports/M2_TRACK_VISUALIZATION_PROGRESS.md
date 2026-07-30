# M2 track visualization progress report

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: accepted by cloud validation

## Outcome

The M2 tracker can now drive the existing annotated-video and WebSocket JPEG outputs. Operator imagery uses stable
track identities and explicitly distinguishes detector-backed observations from finite motion-model predictions.

## Implemented

- Deterministic identity colors and `ID`, class, lifecycle, observation, and confidence labels.
- Solid boxes and trail segments for observed states.
- Orange dashed boxes and trail segments for inferred states.
- Configurable, bounded per-track trail history with no video-length memory growth.
- Automatic history reset on source changes or frame-sequence rollback.
- Fail-fast rejection when a track state does not belong to the rendered frame.
- Track-aware annotated video without changing the existing detection-only behavior when tracking is disabled.
- Track-aware WebSocket JPEG preview and compact per-track metadata for operator UI consumers.
- Shared rendering behavior between recorded video and live preview.

## Cloud verification gate

The focused `m2-tracking` workflow now includes visualization coverage for trail rendering, inferred styling,
frame-alignment validation, preview JPEG encoding, metadata publication, and configuration validation. The ROADMAP
MOT visualization task remains open because metric evaluation and dataset-level visual review are not yet complete.

## Cloud validation result

- Normal CI run [`30523130115`](https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30523130115)
  passed quality checks and the Python 3.10/3.12 matrix.
- Cloud validation run [`30523130081`](https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30523130081)
  passed the M2 tracking/visualization suite, M1 replay suite, and original-weight CPU regression.
- The ROADMAP combined MOT evaluation/visualization item remains open until metric orchestration and dataset-level
  visual review are complete.

## Next increment

1. Add MOT ground-truth adapters and evaluator orchestration.
2. Report HOTA, DetA, AssA, IDF1, ID switches, and fragmentation.
3. Add offline HTML/video error review for ID switches, losses, recoveries, and inferred intervals.
