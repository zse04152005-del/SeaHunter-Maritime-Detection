# Risk register

Last reviewed: 2026-07-30

| ID | Risk | Impact | Current control | Status / exit condition |
|---|---|---|---|---|
| R-001 | Legacy Ultralytics source is AGPL-3.0 while the repository root is MIT | Closed-source distribution may be non-compliant | Pinned baseline boundary and third-party notice | Open until commercial license or verified framework migration |
| R-002 | Real SeaDronesSee/maritime data is absent | Detection, tracking, weather, and ranging quality claims cannot be reproduced | Synthetic tests are labeled `dataset_claim: false`; data gates remain open | Open until versioned leakage-free dataset is available |
| R-003 | No NVIDIA/Jetson target is attached | NVDEC, TensorRT, power, thermal, and soak gates cannot be accepted | ONNX/CPU checks run in cloud; hardware claims are prohibited | Open until target-device validation |
| R-004 | Small/blurred targets can produce unreliable ReID embeddings | Identity hijacking and false recovery | ReID defaults off; size/visibility/overlap/brightness/sharpness gates and tracklet templates are implemented | Open until gates and encoder are calibrated on held-out real maritime crops |
| R-005 | Sea texture and waves can corrupt visual GMC | Incorrect warp can increase ID switches | Foreground masking, RANSAC support, motion plausibility gates, identity fallback | Monitoring; validate by sea state on real video |
| R-006 | Telemetry/video clock or frame transforms can be wrong | Plausible-looking but false absolute locations and TTC | Accepted calibration/sync/transform gates, uncertainty, and explicit relative-only fallback | Open until RTK/AIS/rangefinder truth-set validation |
| R-007 | Long-running edge services can leak resources or fail recovery | Loss of monitoring and alerts | Bounded queues, reconnect, O(1) soak accounting, degradation, canary rollback, and fault matrix | Open until 8 h/72 h and live-fault target-device tests |
| R-008 | Maritime video may contain identifiable people and precise locations | Privacy, safety, and regulatory harm | Data card, access/retention requirements, evidence audit, and independent privacy gate | Open until site-specific privacy approval |
| R-009 | Engineering fixtures could be mistaken for field acceptance | Unsafe or misleading V1 release | Reports label synthetic/template evidence; release freeze requires sea trial, reviews, real data, hardware, and 72 h gates | Open until immutable V1 manifest is legitimately frozen |

Risks are not closed by synthetic evidence. Each closure must link a dataset/device version, Git commit, report, and
reviewer decision.
