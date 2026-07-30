# SeaHunter-VIS

SeaHunter-VIS is the system-level evolution of the original SeaHunter maritime detector. The target is a video-first maritime UAV perception platform covering tiny-object detection, multi-object tracking, trajectory recovery, re-identification, geo-ranging, intrusion alerts, edge inference, and degradation robustness.

## Current status

The repository has completed the locally verifiable M0 work and has started **M1: video-stream detection MVP**. Dataset metric reproduction and TensorRT validation remain gated by external data and NVIDIA target hardware.

- The original detector and weight are retained as a reproducible legacy baseline.
- Historical training and inference scripts live in `legacy/`.
- Historical paper sources live in `docs/archive/`.
- New system code is developed under `src/seahunter/`.
- M1 now includes file/device/RTSP/SRT/HTTP ingestion, decode timestamps, live-source reconnection, a bounded
  freshness-first worker, deterministic JSONL replay, and FPS/P50/P95/P99 performance summaries.
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
uv pip install -e ".[dev,video]"
python -m unittest discover -s tests -v
python evaluation/inspect_baseline.py
```

Install the optional legacy detector stack only when running model tests:

```powershell
uv pip install -e ".[legacy-detector]"
```

## Video replay

Offline replay processes every frame and keeps runtime timing outside the deterministic JSONL metadata:

```powershell
seahunter-video-replay .\samples\flight.mp4 `
  --output .\outputs\flight.jsonl `
  --device cpu
```

For a live source, enable the bounded latest-frame pipeline. Queue pressure drops stale frames instead of allowing
latency and memory to grow without bound:

```powershell
seahunter-video-replay rtsp://camera.example/live `
  --output .\outputs\live.jsonl `
  --realtime `
  --buffer-capacity 2 `
  --device 0
```

Use `--max-reconnect-attempts -1` for an always-on service. The default finite retry budget is safer for CLI jobs.
Hardware decoding is requested through the selected OpenCV backend when available, but NVDEC must still be verified
on the target NVIDIA device before it is treated as an accepted deployment capability.

## Important licensing note

The original repository root is MIT licensed, but the temporarily vendored Ultralytics source declares AGPL-3.0. The root MIT license does not replace third-party licensing obligations. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the framework decision record before any commercial or closed-source distribution.

## Baseline model

- Weight: `weights/seahunter_best.pt`
- SHA-256: `d8ff23bdcd3a707b174ee2ad6274518c46cc0108596bfbfd673fe0535cbeb28f`
- Architecture: P2/P3/P4/P5 detector with SPDConv, MultiSEAM, EMA, and CIoU/NWD loss blending
- Framework baseline: Ultralytics 8.3.234

The historical README and modification notes are preserved in `docs/archive/`.
