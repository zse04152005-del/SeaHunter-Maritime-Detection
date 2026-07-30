# Engineering change log

## 2026-07-30

- Froze the original detector, custom source, weight hash, framework version, and licensing boundary.
- Restored the incomplete local Ultralytics data package and made local-source activation deterministic.
- Connected configurable CIoU/NWD loss blending and validation.
- Reorganized historical code and paper assets under `legacy/` and `docs/archive/`.
- Added locked packaging, Ruff, strict mypy, GitHub Actions, legacy-weight CPU smoke testing, and ONNX consistency.
- Added bounded video ingestion, replay, preview, Parquet/JSONL outputs, monitoring, and reconnect behavior.
- Added ByteTrack, track audit/MOT output, visualization, TrackEval evaluation, BoT-SORT visual GMC, and cloud ablations.
- Replaced the legacy single-entry pseudo-ablation with a strict multi-variant, three-seed experiment plan and dry-run
  validation path.
- Added auditable OpenCV decoder build/selection reports and deterministic hardware-to-software fallback coverage;
  target-device NVDEC acceptance remains explicitly open.
- Added bounded IMU/gimbal motion priors, visual/prior affine fusion, per-track fusion audit fields, and a
  `FrameResult` telemetry integration seam without making absolute-position or real-data benefit claims.
- Added pre-assignment Kalman motion gating, high/low/new association audit fields and counters, schema 4 migration
  guidance, and a focused cloud-validation job.
- Added quality-gated BoT-SORT ReID, clear-frame tracklet template aggregation, explicit tiny-target bypass, a
  replaceable appearance encoder contract, and focused identity-recovery tests.
- Cloud-accepted the M2 ReID increment with 19 focused tests while retaining the real-maritime calibration gate.
- Added finite extrapolation, bounded short-gap interpolation, schema 5 inference-method audit, CLI controls, and a
  focused cloud-validation suite.
- Cloud-accepted M2 trajectory recovery with 32 focused tests; real maritime occlusion metrics remain gated.
- Added strict camera calibration, MAVLink aggregation, affine clock synchronization, frame/telemetry latency
  alignment, and explicit camera/FRD/NED/ENU/WGS84 transforms for the M3 foundation.
- Cloud-accepted the M3 calibration/telemetry foundation with 22 focused tests and a machine-readable audit artifact.
- Added sea-plane target geolocation, uncertainty propagation, ENU EKF smoothing, relative motion classes, and
  quality-gated TTC with explicit degradation reasons.
- Cloud-accepted M3 geolocation wiring with 21 focused tests while retaining the real truth-set accuracy gate.
- Added local/geodetic danger zones, six stateful rules, hysteresis/dwell/cooldown, reliability gating, weighted risk,
  and acknowledged event lifecycle for M4.
- Cloud-accepted the M4 event core with 18 focused tests; real event-level performance gates remain open.
- Added restart-safe SQLite event persistence, atomic evidence bundles, REST/WebSocket/MQTT publication, operator
  acknowledgement/feedback, and false-positive hard-sample export; cloud-accepted the M4 closure with 16 tests.
- Added M5 data contracts, leakage/weather/size/hard-negative audits, active learning, temporal Tracklet pseudo-label
  filters, confidence calibration, DVC/lakeFS configuration, and MLflow provenance; cloud-accepted 11 tests.
- Added M6 selective ROI, finite temporal monitoring, bounded joint confidence, export-friendly joint downsampling,
  clip-stable weather routing, and three-seed ablation contracts; cloud-accepted 11 tests.
- Added M7 export/build specifications, signed model packages, thermal/memory/timeout degradation, durable canary
  rollback, soak/fault gates, Jetson/DeepStream templates, and 15 productization tests.
- Fixed dynamic ONNX export in the custom EMA module with equivalent axis reductions; cloud-exported and verified
  static and dynamic-batch ONNX artifacts against PyTorch.
- Added M8 sea-trial coverage/metric/evidence/review audits and a release freeze that refuses incomplete external
  gates; cloud-accepted 8 readiness tests while intentionally leaving V1 unfrozen.

This log summarizes accepted engineering increments. Exact implementation and validation evidence remains in Git
history and `docs/reports/`.
