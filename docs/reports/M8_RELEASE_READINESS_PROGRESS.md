# M8 sea-trial and release-readiness progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: release-readiness controls cloud accepted; V1 is intentionally not frozen

## Implemented

- Machine-readable sea-trial plan covering independent locations, time bands, altitude bands, five weather
  conditions, sea states, and minimum blind-operator trials.
- Per-trial calibration, telemetry synchronization, zone configuration, event audit, evidence bundle, and exact
  signed model-package references.
- Per-trial false alarms/hour, missed alerts, reacquisition, and distance MAE gates; one failed condition blocks the
  submission instead of being hidden in an aggregate.
- Independent named safety, privacy, license, and operations review requirements.
- External truth-set, real metric, TensorRT, NVDEC, 72-hour soak, and live-fault gates.
- Atomic V1 release-manifest freeze that refuses to write on any incomplete sea-trial, review, data, or hardware gate.
- Sea-trial protocol, operator guide, compliance checklist, release checklist, and V2 backlog.

## Cloud acceptance

- Accepted implementation commit: `524e1239e78a309148605b36158ceb1583dc3222`.
- The `m8-readiness` job collected and passed 8 acceptance/refusal-path tests.
- Cloud validation: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30545206469>
- Focused job: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30545206469/job/90879501692>
- JUnit and deliberately failing readiness-template report:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30545206469/artifacts/8760389755>
- Python 3.10/3.12 CI and static gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30545206643>

## Why V1 remains unfrozen

No real sea-trial submission, independent operator blind test, formal privacy/license/safety/operations approval,
M3 external truth set, target TensorRT/NVDEC evidence, live fault matrix, or 72-hour target-device result exists in
the workspace. The committed JSON is visibly marked as a template and the generated readiness artifact fails by
design. Freezing V1 before those conditions would be a false acceptance claim, so the release gate correctly blocks
it.
