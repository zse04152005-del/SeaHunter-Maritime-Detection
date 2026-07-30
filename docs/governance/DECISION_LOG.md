# Decision log

| Date | ID | Decision | Reason | Revisit trigger |
|---|---|---|---|---|
| 2026-07-30 | D-001 | Retain pinned Ultralytics 8.3.234 only as the reproducible legacy boundary | The original weight depends on custom parser/modules | Production licensing or framework migration decision |
| 2026-07-30 | D-002 | Keep new video/tracking/geometry/event code framework-neutral | Avoid coupling the product system to legacy detector types | Only if a stable cross-framework contract is insufficient |
| 2026-07-30 | D-003 | Use ByteTrack as the first tracker and add the motion-only BoT-SORT branch before ReID | Establish auditable motion baselines before appearance complexity | ReID quality gates pass on real maritime crops |
| 2026-07-30 | D-004 | Use official TrackEval pinned by commit, with a narrow NumPy compatibility bridge | PyPI constraints conflict with the verified NumPy range | Upstream release becomes compatible and is revalidated |
| 2026-07-30 | D-005 | Treat synthetic metrics as wiring checks, never dataset performance claims | Prevent false acceptance without maritime truth data | Never; real claims require versioned held-out data |
| 2026-07-30 | D-006 | Run tests/experiments in GitHub Actions when possible | Keep the local workstation for editing and static checks | Target hardware tasks require a dedicated runner |
| 2026-07-30 | D-007 | Replace the legacy one-entry “ablation” script with a strict JSON experiment matrix | Freeze variants, seeds, NWD controls, and manifests before GPU allocation | Schema migration with backward-compatibility note |
| 2026-07-30 | D-008 | Separate decoder build capability, actual source selection, and NVDEC target proof | Generic OpenCV hardware flags can silently fall back and cannot independently prove NVDEC | A dedicated runtime exposes stronger vendor-specific decoder evidence |
| 2026-07-30 | D-009 | Use aligned IMU/gimbal rotation only as a gated GMC prior | Telemetry can rescue weak visual texture but cannot replace visual evidence or calibrated translation geometry | M3 calibration/time-sync results or real-sequence disagreement analysis |
| 2026-07-30 | D-010 | Gate high/low ByteTrack association with Kalman Mahalanobis distance before Hungarian assignment | IoU alone can match physically implausible center/scale jumps, and post-assignment rejection can hide valid alternatives | Held-out maritime MOT calibration supports a different confidence level or measurement model |
| 2026-07-30 | D-011 | Permit ReID only after crop quality gates and aggregate only eligible observations into tracklet templates | Tiny, blurred, clipped, dark, saturated, or overlapping maritime crops can cause identity hijacking | Real crop calibration changes gates or a validated ONNX ReID encoder replaces the histogram baseline |
| 2026-07-30 | D-012 | Keep live recovery causal and restrict hindsight interpolation to a bounded offline-output buffer | Retrospective boxes must not leak into real-time alerts or be mislabeled as observations | A validated fixed-lag smoother replaces linear interpolation with equivalent audit separation |

Architecture decisions with broader context remain under `docs/adr/`.
