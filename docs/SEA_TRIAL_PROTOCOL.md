# Sea-trial and operator blind-test protocol

## Before deployment

Freeze the Git commit, signed model package, TensorRT engine, edge image, data/calibration versions, camera
calibration, telemetry offset/drift report, zone configuration, metric thresholds, and trial plan. Verify permits,
weather limits, emergency stop, privacy notice, retention, and named safety lead. Never tune thresholds on the frozen
test voyages.

## Coverage and execution

Use at least the approved number of independent locations and cover dawn/day/dusk/night, low/medium/high altitude,
calm/moderate/rough sea, and normal/low-light/fog/glare/high-wave conditions. Each trial records source identity,
timestamps, device/software versions, health metrics, dropped/reconnected frames, operator actions, events, evidence
bundles, and ground truth. Operators in blind tests must not know injected-event timing or expected labels.

Abort on unsafe flight conditions, privacy boundary violation, critical temperature, repeated inference timeout,
unrecoverable stream/process failure, invalid calibration/synchronization, or unsigned package. Record an aborted
trial; do not silently replace it.

## Scoring and closure

Report false alarms/hour, missed alerts, event delay, ID switches, reacquisition, distance MAE/relative error,
evidence completeness, acknowledgement latency, FPS/P95/power/temperature/VRAM, and failure recovery by condition.
Every issue has an owner, severity, corrective commit/package, retest evidence, and closure approval. Run
`seahunter-acceptance-audit` on the signed submission. Only the release-gate code may create the final frozen manifest,
and only after all sea-trial, review, truth-set, TensorRT/NVDEC, fault, and 72-hour gates pass.
