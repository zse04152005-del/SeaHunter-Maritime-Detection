# M7 ONNX, TensorRT, and edge productization progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: CPU/cloud productization gates accepted; NVIDIA target-device gates remain open

## Implemented

- Explicit static and dynamic-batch ONNX export specifications and TensorRT min/opt/max build profiles.
- Real-only, versioned, all-weather INT8 calibration gate that rejects synthetic-only or incomplete coverage.
- DeepStream/NVDEC/TensorRT/CUDA-stream/preallocated-buffer configuration and capability reporting without treating
  requested acceleration as proof of effective acceleration.
- Thermal, GPU-memory, latency, and timeout degradation from normal to reduced ROI, lower resolution, and safe stop,
  with healthy recovery hysteresis.
- Signed ZIP model packages containing the complete base `ModelManifest`, preprocessing, backend, precision, input
  shapes, calibration reference, evaluation metrics, and per-artifact SHA-256. The signer interface supports an
  HSM/KMS implementation; HMAC is explicitly limited to isolated site testing.
- Durable staged/canary/stable/rolled-back release state with atomic persistence and automatic candidate rollback.
- O(1)-memory soak accumulator, memory-slope/duration/fatal-error gates, and complete network/stream/corrupt-frame /
  clock-drift/process-restart evidence contract.
- Jetson container template and deployment, upgrade, rollback, degradation, soak, and fault runbook.

## Cloud acceptance

- Productization implementation commit: `2b170e295a8a4d833a2bfa06dbaf742b5e6ace0e`.
- `m7-productization` collected and passed 15 tests:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30543452932/job/90873571689>
- Productization JUnit artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30543452932/artifacts/8759669247>
- Python 3.10/3.12 CI and static gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30543453321>

## Static and dynamic ONNX closure

The first dynamic-batch export correctly failed at the custom EMA module's axis-preserving adaptive pooling. Commit
`cb20e55c62fa796ce99a2b38c8c3206b2925256a` replaces it with mathematically equivalent axis `ReduceMean`, retaining
weight compatibility and supporting dynamic dimensions. The cloud rerun exported both graphs and both passed the
frozen PyTorch/ONNX maximum/mean absolute-error thresholds.

- Accepted ONNX run: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30544295513>
- Focused job: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30544295513/job/90876428897>
- Static/dynamic ONNX files and consistency reports (about 164 MB):
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30544295513/artifacts/8760040796>

## Remaining hardware gates

No NVIDIA target is attached. TensorRT FP16/INT8 engines, PyTorch/ONNX/TensorRT consistency, real NVDEC selection,
CUDA overlap, target FPS/P95/power/temperature/VRAM, DeepStream image operation, live fault recovery, 8-hour
performance, and 72-hour stability remain blocked. Cloud-generated ONNX results and time-compressed soak unit tests
must not be reported as those device results.
