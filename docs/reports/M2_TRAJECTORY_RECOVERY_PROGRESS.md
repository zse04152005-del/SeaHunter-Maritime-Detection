# M2 trajectory recovery progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; real maritime occlusion evaluation remains open

## Implemented

- Finite causal Kalman extrapolation bounded by `max_lost_frames`, with confidence decay and covariance growth.
- A bounded-latency output writer that converts a recovered short gap from extrapolated boxes to interpolation
  between its two observed endpoints.
- Strict monotonic-frame and unique-track validation plus deterministic frame/identity output order.
- Separate `observed`/`inferred` evidence and `extrapolated`/`interpolated` inference methods in audit schema 5.
- Conservative uncertainty: interpolation retains the causal Kalman covariance and never upgrades confidence.
- CLI/configuration controls and sequence counters for interpolated gaps/points and remaining extrapolated points.

## Safety and acceptance boundary

Interpolation affects offline track JSONL/MOT outputs only and introduces a bounded delay. Live preview, alerting,
and tracker association remain causal. It does not bridge gaps longer than the configured horizon, invent missing
identities, or treat inferred boxes as detector observations. Synthetic constant-motion tests validate contracts and
buffer behavior only; real occlusion recovery benefit still requires a held-out maritime MOT set stratified by
occlusion duration and target size.

## Cloud acceptance

- Accepted commit: `509374aa99518b13c87a40922a29fe07f321bb8b`.
- The focused `m2-recovery` job passed 32 tests in 1.53 seconds.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30537143811>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30537143737>
- Trajectory-recovery artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30537143811/artifacts/8757120539>

The first cloud attempt exposed a missing OpenCV workflow extra before tests were collected. Commit `509374a`
corrected the job environment; only the successful rerun above is accepted.

## Remaining external gate

- Measure fragmentation, recovery rate, localization error, and identity errors by occlusion duration and target-size
  bucket on a leakage-free maritime MOT test set.
