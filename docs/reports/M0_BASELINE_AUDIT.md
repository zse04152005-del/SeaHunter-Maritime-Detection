# M0 baseline audit

## Repository snapshot

- Upstream repository: `zse04152005-del/SeaHunter-Maritime-Detection`
- Original branch: `main`
- Original HEAD: `c7cb6b5`
- Upgrade integration branch: `develop`
- Original commit count: 2

## Preserved baseline

- Weight: `weights/seahunter_best.pt`
- Size: 64,096,354 bytes
- SHA-256: `d8ff23bdcd3a707b174ee2ad6274518c46cc0108596bfbfd673fe0535cbeb28f`
- Ultralytics version: 8.3.234
- Model: four detection levels P2/P3/P4/P5
- Custom modules: SPDConv, EMA, MultiSEAM
- Bbox objective: 0.5 CIoU + 0.5 NWD similarity loss, followed by the standard box gain

## Confirmed findings

1. The original `run_ablation.py` contained one experiment rather than a real ablation matrix.
2. The training script imported `ultralytics` without ensuring the repository-local fork was selected.
3. The inference script inserted the local source path but rejected RTSP/SRT URLs with a local `Path.exists()` check.
4. Inference did not request streaming results, which is unsuitable for unbounded video.
5. NWD is called from `BboxLoss.forward`; it is not dead code in the cloned repository.
6. The NWD blend and normalization constant were hard-coded and had no ablation switch.
7. MultiSEAM uses `exp(sigmoid(gate))`, which only amplifies features and cannot directly attenuate a channel.
8. EMA only checks `c1 // groups > 0`; it should also require `c1 % groups == 0` for general use.
9. The copied Ultralytics package lacked the required `ultralytics/data` directory.
10. The legacy requirements installed both `opencv-python` and `opencv-python-headless` and did not lock a reproducible environment.
11. The vendored framework declares AGPL-3.0 despite the root MIT license.

## Cleanup completed

- Moved historical scripts into `legacy/`.
- Moved paper source into `docs/archive/paper_visualizations/`.
- Removed 12 generated PNG files totaling approximately 53.23 MiB; they remain recoverable from Git history and can be regenerated from archived scripts.
- Retained the baseline weight and custom detector source.
- Restored the missing `ultralytics/data` package from upstream tag `v8.3.234`.

## Current limitations

- A project-local Python 3.12 environment is available with PyTorch 2.13.0 CPU, Ultralytics 8.3.234, and locked dependencies.
- GPU/CUDA and TensorRT validation are not yet complete.
- Original dataset files are not present, so the published detection metrics cannot yet be independently reproduced.
- The production framework licensing choice remains open.

## Verification completed

- Static baseline integrity audit passes, including the weight hash and NWD call path.
- The repository-local Ultralytics package is selected instead of the installed package.
- The original weight loads as a five-class `DetectionModel`.
- A blank-frame CPU inference smoke test passes.
- SPDConv, EMA, MultiSEAM forward/backward tests pass.
- NWD similarity, parameter validation, RTSP/SRT parsing, schemas, and geofence tests pass.
- Current framework-neutral result after the M1 edge-output increment: 48 tests passed.
- Ruff formatting/linting and strict mypy checks pass for new system code.
- Static ONNX export at 256×256 passes with output shape `[1, 9, 5440]`.
- PyTorch/ONNX raw-output consistency passes: max absolute error `1.8310546875e-4`, mean absolute error `2.9645066206e-6`.

## M0 exit work still required

- The configuration-driven multi-variant experiment matrix is implemented; cloud dry-run acceptance is recorded in
  `docs/reports/M0_GOVERNANCE_PROGRESS.md`.
- Produce TensorRT consistency and latency results on the target NVIDIA hardware.
