# M6 deployable innovation progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: engineering contracts cloud accepted; real ablations and hardware metrics remain open

## Implemented

- Selective high-resolution planning from objectness, motion, active/lost tracks; Top-K ranking, padding, overlap
  merge, periodic global refresh, cross-scale class-aware deduplication, and latency/queue/thermal budget reduction.
- Finite temporal feature history with spatial/change/motion fusion and explicit camera-motion, periodic-wave, and
  target-candidate classes. State resets on new source, stream gap, scene change, and feature-shape change.
- Temperature-scaled detection/tracking confidence fusion with motion, optional ReID, covariance uncertainty,
  bounded persistence boost, and periodic-wave confidence veto.
- Fixed low-pass plus Laplacian-energy reference downsampling using only real `Conv/Abs/Mul/Add`-equivalent
  primitives; no FFT, complex number, or TensorRT plugin requirement.
- Clip-voted weather routing with confidence gate, minimum dwell, a single robust model by default, preprocessing /
  threshold profiles first, explicit expert opt-in, and a real-coverage gate before Sim2Real.
- Unified three-seed ablation schema for precision, recall, false alarms/hour, parameters, FLOPs, VRAM, and TensorRT
  latency, including mean/standard deviation and exportability.

## Cloud acceptance

- Accepted implementation commit: `9ae6fda831222f21c17ada4e1613da7a0ca37cd6`.
- The `m6-innovation` job collected and passed 11 tests.
- Cloud validation: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30542613242>
- Focused job: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30542613242/job/90870759209>
- JUnit artifact: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30542613242/artifacts/8759331366>
- Python 3.10/3.12 CI and static gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30542610822>

## Acceptance boundary

The tests prove deterministic control logic, bounded state, fusion/veto behavior, and operator manifests. They do not
prove learned accuracy or edge speed. Remaining gates require a frozen real maritime dataset, GPU training for at
least three seeds, full-frame/fixed-tile/ROI and single-frame/temporal comparisons, hard-negative slices, ONNX and
TensorRT export of the trained graph, and measured latency/VRAM/thermal performance on the target NVIDIA device.
