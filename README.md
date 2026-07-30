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
- The ByteTrack checkbox in the roadmap remains open until the feature branch passes GitHub Actions cloud validation.
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
uv pip install -e ".[dev,video,edge,parquet,tracking]"
python -m unittest discover -s tests -v
python evaluation/inspect_baseline.py
```

Install the optional legacy detector stack only when running model tests:

```powershell
uv pip install -e ".[legacy-detector]"
```

The authoritative test matrix runs in GitHub Actions on Python 3.10 and 3.12. The manual `cloud-validation`
workflow provides focused `legacy-detector`, `m1-replay`, `m2-tracking`, and `all` suites with JUnit artifacts.

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

## ByteTrack baseline

Enable the M2 tracker during replay and write both evaluation and audit outputs:

```powershell
seahunter-video-replay .\samples\flight.mp4 `
  --output .\outputs\flight.jsonl `
  --tracker bytetrack `
  --tracks-jsonl .\outputs\flight-tracks.jsonl `
  --mot-output .\outputs\flight-mot.txt `
  --track-frame-rate 25 `
  --track-emit-lost `
  --device cpu
```

MOT output contains observed states by default. Add `--mot-include-inferred` only when the downstream evaluator is
intended to consume motion-model predictions. The audit JSONL always retains `observed`/`inferred`, lifecycle,
association score, covariance, age, time-since-update, tracker configuration ID, and loss reason. Starting parameters
are recorded in `configs/tracking/bytetrack.maritime.yaml` and must be calibrated on video-level maritime data.

## Important licensing note

The original repository root is MIT licensed, but the temporarily vendored Ultralytics source declares AGPL-3.0. The root MIT license does not replace third-party licensing obligations. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the framework decision record before any commercial or closed-source distribution.

## Baseline model

- Weight: `weights/seahunter_best.pt`
- SHA-256: `d8ff23bdcd3a707b174ee2ad6274518c46cc0108596bfbfd673fe0535cbeb28f`
- Architecture: P2/P3/P4/P5 detector with SPDConv, MultiSEAM, EMA, and CIoU/NWD loss blending
- Framework baseline: Ultralytics 8.3.234

The historical README and modification notes are preserved in `docs/archive/`.
