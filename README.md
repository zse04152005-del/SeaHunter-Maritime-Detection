# SeaHunter-VIS

SeaHunter-VIS is the system-level evolution of the original SeaHunter maritime detector. The target is a video-first maritime UAV perception platform covering tiny-object detection, multi-object tracking, trajectory recovery, re-identification, geo-ranging, intrusion alerts, edge inference, and degradation robustness.

## Current status

The repository has completed the locally verifiable M0 work and implemented the M1 video-stream foundation. It has
now started **M2: multi-object tracking and trajectory recovery**. Dataset metric reproduction and TensorRT
validation remain gated by external data and NVIDIA target hardware.

- The original detector and weight are retained as a reproducible legacy baseline.
- Historical training and inference scripts live in `legacy/`.
- Historical paper sources live in `docs/archive/`.
- New system code is developed under `src/seahunter/`.
- M1 now includes file/device/RTSP/SRT/HTTP ingestion, decode timestamps, live-source reconnection, a bounded
  freshness-first worker, deterministic JSONL/Parquet replay, annotated video, WebSocket JPEG preview, and
  CPU/memory/NVIDIA device performance summaries.
- The M2 baseline adds framework-neutral two-stage ByteTrack association, Kalman motion prediction, explicit track
  lifecycle/loss reasons, optional short-occlusion predictions, MOTChallenge export, and auditable track JSONL.
- A selectable BoT-SORT motion baseline adds foreground-masked sparse optical flow, RANSAC affine camera-motion
  compensation, plausibility/quality fallback gates, and per-frame GMC audit metadata. Appearance ReID remains off
  until the separate maritime small-target quality gate is implemented.
- The BoT-SORT/GMC implementation passed the Python 3.10/3.12 matrix and focused cloud ablation; its synthetic
  ID-switch improvement is a wiring check, not a real maritime performance claim.
- The ByteTrack baseline passed the Python 3.10/3.12 CI matrix plus focused M2 tracking, M1 replay, and legacy-weight
  CPU cloud validation suites.
- The complete execution order and acceptance gates are defined in [ROADMAP.md](ROADMAP.md).

## Repository layout

```text
configs/        Versioned data, model, tracker, and runtime configuration
docs/           Architecture decisions, reports, and historical material
evaluation/     Offline replay and metric tools
legacy/         Original project entry points retained for reproducibility
src/seahunter/  Framework-neutral system implementation
tests/          Unit, integration, replay, and export tests
training/       Configuration-driven training entry points
weights/        Legacy baseline weight during M0
yolo_source/    Temporary pinned Ultralytics 8.3.234 baseline fork
```

## Development setup

The target development Python is 3.10 or 3.11. Python 3.12 may be used for framework-neutral tests, but CUDA, TensorRT, and Jetson environments will be pinned separately.

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,video,edge,parquet,tracking,evaluation]"
python -m unittest discover -s tests -v
python evaluation/inspect_baseline.py
```

Install the optional legacy detector stack only when running model tests:

```powershell
uv pip install -e ".[legacy-detector]"
```

The authoritative test matrix runs in GitHub Actions on Python 3.10 and 3.12. The manual `cloud-validation`
workflow provides focused `m0-governance`, `legacy-detector`, `m1-replay`, `m2-tracking`, `m2-evaluation`, `m2-gmc`,
and `all` suites with JUnit and metric artifacts.

## Detector experiment planning

Validate and expand the frozen CIoU/NWD detector ablation before allocating a GPU:

```powershell
python training/run_experiments.py `
  --config configs/experiments/m0_detector_nwd_ablation.json `
  --dry-run
```

The strict configuration expands CIoU-only and CIoU/NWD variants over three fixed seeds, rejects unknown keys,
duplicate variants, invalid loss controls, and repository-escaping paths, and records a SHA-256 of the experiment
source. Removing `--dry-run` requires the legacy detector stack plus the real dataset and writes a per-run manifest
with the Git commit, dependency version, device, seed, and complete resolved controls. Dataset-free CI validates the
plan only and never reports training quality.

## Video replay

Offline replay processes every frame and keeps runtime timing outside the deterministic JSONL metadata:

```powershell
seahunter-video-replay .\samples\flight.mp4 `
  --output .\outputs\flight.jsonl `
  --parquet .\outputs\flight.parquet `
  --annotated-video .\outputs\flight-annotated.mp4 `
  --device cpu
```

For a live source, enable the bounded latest-frame pipeline. Queue pressure drops stale frames instead of allowing
latency and memory to grow without bound:

```powershell
seahunter-video-replay rtsp://camera.example/live `
  --output .\outputs\live.jsonl `
  --realtime `
  --buffer-capacity 2 `
  --preview `
  --device 0
```

The preview page defaults to `http://127.0.0.1:8000` and exposes `/health`, `/metrics`, and `/ws/preview`. The
single-slot JPEG hub retains only the latest frame, so a slow browser cannot create an unbounded server queue.

The edge-service entry point enables real-time buffering, preview, and unlimited stream reconnection by default:

```powershell
seahunter-edge-service rtsp://camera.example/live `
  --output .\outputs\edge-live.jsonl `
  --annotated-video .\recordings\edge-live.mp4 `
  --device 0
```

The server binds to loopback by default. Set `--preview-host 0.0.0.0` only on a controlled edge network with the
appropriate firewall and access controls.

Use `--max-reconnect-attempts -1` for an always-on service. The default finite retry budget is safer for CLI jobs.
Hardware decoding is requested through the selected OpenCV backend when available, but NVDEC must still be verified
on the target NVIDIA device before it is treated as an accepted deployment capability.

Generate a machine-readable OpenCV build report without opening media:

```powershell
seahunter-decode-capabilities --output .\reports\decoder-build-capabilities.json
```

On the target edge device, add `--source` to open and decode one frame. The report records the requested and selected
backend, the effective acceleration API, and any deterministic software fallback. It never treats OpenCV's generic
hardware flag or CUDA visibility as proof of NVDEC, and it does not serialize the source URL or credentials.

```powershell
seahunter-decode-capabilities `
  --source rtsp://camera.example/live `
  --backend ffmpeg `
  --output .\reports\target-decoder-selection.json
```

## ByteTrack and BoT-SORT baselines

Enable the M2 tracker during replay and write both evaluation and audit outputs:

```powershell
seahunter-video-replay .\samples\flight.mp4 `
  --output .\outputs\flight.jsonl `
  --tracker bytetrack `
  --tracks-jsonl .\outputs\flight-tracks.jsonl `
  --mot-output .\outputs\flight-mot.txt `
  --track-frame-rate 25 `
  --track-emit-lost `
  --track-trail-length 30 `
  --device cpu
```

MOT output contains observed states by default. Add `--mot-include-inferred` only when the downstream evaluator is
intended to consume motion-model predictions. The audit JSONL always retains `observed`/`inferred`, lifecycle,
association score, covariance, age, time-since-update, tracker configuration ID, and loss reason. Starting parameters
are recorded in `configs/tracking/bytetrack.maritime.yaml` and must be calibrated on video-level maritime data.

Use the BoT-SORT motion branch for a moving UAV camera:

```powershell
seahunter-video-replay .\samples\flight.mp4 `
  --output .\outputs\flight.jsonl `
  --tracker botsort `
  --tracks-jsonl .\outputs\flight-tracks.jsonl `
  --mot-output .\outputs\flight-mot.txt `
  --gmc-downscale 2 `
  --gmc-minimum-inliers 12 `
  --device cpu
```

The visual GMC estimator masks detector boxes, tracks background corners with pyramidal LK flow, fits a partial
affine transform with RANSAC, and rejects low-support or implausible translation/scale/rotation estimates. Rejected
frames use an identity fallback instead of moving tracks with an unreliable warp. Track audit JSONL schema v2
records the affine coefficients, quality, applied flag, and fallback reason. `--gmc-disabled` provides a paired
ablation with otherwise identical BoT-SORT settings; starting parameters are versioned in
`configs/tracking/botsort.maritime.yaml`.

The included synthetic camera-pan experiment validates wiring and known ID-switch behavior only. It is not evidence
of real maritime tracking quality; acceptance on real video still requires paired no-GMC/GMC TrackEval results on
the same leakage-free moving-camera sequences.

When tracking is enabled, annotated video and WebSocket JPEG preview switch from raw detection boxes to stable track
IDs and bounded trails. Observed states use identity colors and solid boxes; motion-model predictions use orange
dashed boxes and trail segments. Preview metadata includes the same lifecycle and observation fields for operator UI
and audit consumers.

## MOT evaluation

Evaluate one MOTChallenge sequence with the standard TrackEval HOTA, CLEAR MOT, and identity metrics:

```powershell
seahunter-mot-evaluate `
  --ground-truth .\datasets\mot\flight-01\gt.txt `
  --predictions .\outputs\flight-01-mot.txt `
  --sequence-name flight-01 `
  --output .\reports\flight-01.metrics.json `
  --errors-output .\reports\flight-01.errors.json `
  --artifacts-dir .\reports\flight-01.trackeval
```

The evaluator validates positive one-based frame/track IDs, finite positive boxes, and unique identities per frame.
It normalizes maritime classes to a class-agnostic TrackEval sequence and reports HOTA, DetA, AssA, LocA, MOTA,
MOTP, recall, precision, IDF1, ID switches, fragmentation, false positives, and false negatives. The separate error
index identifies the exact frames and identities involved in misses, false positives, ID switches, and fragmented
recoveries for offline video review.

TrackEval is pinned to an exact official MIT-licensed commit. Dataset-level numbers are not accepted until ground
truth and predictions come from a leakage-free video split with documented ignore-region and visibility policy.

## Important licensing note

The original repository root is MIT licensed, but the temporarily vendored Ultralytics source declares AGPL-3.0. The root MIT license does not replace third-party licensing obligations. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the framework decision record before any commercial or closed-source distribution.

## Baseline model

- Weight: `weights/seahunter_best.pt`
- SHA-256: `d8ff23bdcd3a707b174ee2ad6274518c46cc0108596bfbfd673fe0535cbeb28f`
- Architecture: P2/P3/P4/P5 detector with SPDConv, MultiSEAM, EMA, and CIoU/NWD loss blending
- Framework baseline: Ultralytics 8.3.234

The historical README and modification notes are preserved in `docs/archive/`.
