# M5 data loop, robustness, and semi-supervised progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: engineering baseline cloud accepted; real-data improvement gate remains open

## Implemented

- Versioned manifest contracts for video/voyage/location/time, weather, real or synthetic degradation level,
  detection boxes, Track IDs, occlusion, and hard-negative categories.
- Blocking train/validation/test leakage checks at video, voyage, and date/location group level.
- Class long-tail, pixel-size, split, weather, degradation, and wave/wake/bird/foam/reflection reports.
- Active-learning ranking for uncertainty, teacher/student disagreement, track fragmentation, and false alerts with
  deterministic budgets and per-video caps.
- EMA teacher-student baseline configuration with three seeds and frozen-test references.
- Auditable pseudo-label rejection by weather confidence, teacher/student class agreement, temporal IoU, minimum
  Tracklet length, and Tracklet class consistency.
- Dependency-light temperature scaling, ECE, and independent weather-threshold evaluation.
- DVC stage and lakeFS-compatible remote guidance plus MLflow provenance adapter recording Git commit, data version,
  configuration SHA-256, seed, parameters, metrics, and artifacts.
- Data/annotation guide, dataset-card template, and M5 model card.

## Cloud acceptance

- Accepted implementation commit: `6d69202bce4d8c85a89e05ce8116501c762ade7b`.
- The `m5-data` job collected and passed 11 tests.
- Cloud validation: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30541849206>
- Focused job: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30541849206/job/90868251977>
- JUnit and example audit artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30541849206/artifacts/8759010235>
- Python 3.10/3.12 CI and static gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30541849190>

## Acceptance boundary

The committed example manifest contains metadata-only fixtures and proves contracts, leakage detection, and report
wiring. It is not maritime acceptance data. M5 promotion still requires an immutable production data version,
approved privacy/access controls, leakage-free weather-stratified evaluation, and a three-seed teacher-student gain
on the frozen real test set without a material increase in false alarms/hour.
