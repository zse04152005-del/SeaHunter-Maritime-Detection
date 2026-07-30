# Risk register

Last reviewed: 2026-07-30

| ID | Risk | Impact | Current control | Status / exit condition |
|---|---|---|---|---|
| R-001 | Legacy Ultralytics source is AGPL-3.0 while the repository root is MIT | Closed-source distribution may be non-compliant | Pinned baseline boundary and third-party notice | Open until commercial license or verified framework migration |
| R-002 | Real SeaDronesSee/maritime data is absent | Detection, tracking, weather, and ranging quality claims cannot be reproduced | Synthetic tests are labeled `dataset_claim: false`; data gates remain open | Open until versioned leakage-free dataset is available |
| R-003 | No NVIDIA/Jetson target is attached | NVDEC, TensorRT, power, thermal, and soak gates cannot be accepted | ONNX/CPU checks run in cloud; hardware claims are prohibited | Open until target-device validation |
| R-004 | Small/blurred targets can produce unreliable ReID embeddings | Identity hijacking and false recovery | ReID remains disabled; size/quality gate required first | Open; M2 ReID quality-gate task |
| R-005 | Sea texture and waves can corrupt visual GMC | Incorrect warp can increase ID switches | Foreground masking, RANSAC support, motion plausibility gates, identity fallback | Monitoring; validate by sea state on real video |
| R-006 | Telemetry/video clock or frame transforms can be wrong | Plausible-looking but false absolute locations and TTC | Quality-bearing schemas and explicit relative-only fallback | Open; M3 calibration/synchronization gates |
| R-007 | Long-running edge services can leak resources or fail recovery | Loss of monitoring and alerts | Bounded queues, reconnect policy, health/resource metrics | Open until 8 h/72 h target-device soak tests |

Risks are not closed by synthetic evidence. Each closure must link a dataset/device version, Git commit, report, and
reviewer decision.
