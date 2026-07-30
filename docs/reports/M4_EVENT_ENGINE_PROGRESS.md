# M4 danger-zone and event-engine progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: engineering closure cloud accepted; real event-quality calibration remains open

## Implemented

- Validated simple polygons in local ENU metres or `(longitude, latitude)` WGS84 coordinates.
- Local metric projection and signed boundary distance for geodetic zones.
- Enter/exit hysteresis, minimum enter/exit time, minimum dwell time, and per-rule cooldown.
- Enter, exit, dwell, approach, reverse-direction, and TTC collision-trend rules.
- Reliability gate that ignores low-quality frames instead of opening or closing an event on one bad observation.
- Weighted reliability/TTC/class/zone risk score and deterministic event IDs.
- `open/updated/acknowledged/closed` lifecycle with acknowledgement persistence during later updates.

## Service closure

- SQLite latest-state storage with append-only lifecycle transitions and operator-feedback audit records.
- Startup restoration of all non-closed events into the stateful rule engine.
- Bounded pre/post-event JPEG ring with atomic ZIP finalization.
- Evidence bundles contain frames, track JSONL, telemetry JSONL, model manifests, the final event lifecycle state,
  and SHA-256 digests for every bundled artifact.
- REST endpoints for event query, transition history, acknowledgement, and operator feedback.
- Bounded WebSocket history and pluggable MQTT publication. MQTT failures are retained as bounded local diagnostics
  and do not interrupt SQLite persistence, evidence capture, or WebSocket delivery.
- False-positive feedback is appended to a reproducible hard-sample JSONL pool.

## Acceptance boundary

Synthetic paths prove rule state machines and geometry only. Zone accuracy depends on approved mission coordinates,
vertical/horizontal geolocation quality, class mapping, and operator policy. Thresholds and event-level precision,
recall, and false alarms/hour remain real replay and field gates.

## Cloud acceptance

- Accepted commit: `3505d615281b9235f250891aa2cf2a10220f6455`.
- The focused `m4-events` job passed 18 tests in 0.07 seconds.
- Focused cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30539519558>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30539424425>
- M4 event artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30539519558/artifacts/8758075027>

The first run correctly exposed a test fixture that omitted TTC while asserting a TTC-weighted score. The accepted
rerun uses a 30-second TTC above the collision threshold, independently exercising the TTC risk contribution.

## Closure cloud acceptance

- Accepted implementation commit: `6dcd642b22f707267218a7f9360973ff5be00cba`.
- The `m4-closure` job collected and passed 16 persistence, restart, evidence, REST, WebSocket, MQTT, feedback,
  geometry, and lifecycle tests.
- Focused job: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30540735540/job/90864649368>
- JUnit artifact: <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30540735540/artifacts/8758569079>
- Python 3.10/3.12 CI, strict typing, lint, formatting, full test, and original-weight CPU regression:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30540735566>

Synthetic rule paths, fake MQTT clients, and generated JPEG payloads validate contracts and failure handling only;
they do not establish event precision/recall or evidence quality on real maritime video.

## Remaining external gate

- Freeze zone/rule/class policy on leakage-free event replays and report event precision/recall, duplicate rate,
  detection delay, and false alarms/hour by location, weather, and sea state.
