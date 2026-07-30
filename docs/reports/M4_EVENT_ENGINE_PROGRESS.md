# M4 danger-zone and event-engine progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; real event-quality calibration remains open

## Implemented

- Validated simple polygons in local ENU metres or `(longitude, latitude)` WGS84 coordinates.
- Local metric projection and signed boundary distance for geodetic zones.
- Enter/exit hysteresis, minimum enter/exit time, minimum dwell time, and per-rule cooldown.
- Enter, exit, dwell, approach, reverse-direction, and TTC collision-trend rules.
- Reliability gate that ignores low-quality frames instead of opening or closing an event on one bad observation.
- Weighted reliability/TTC/class/zone risk score and deterministic event IDs.
- `open/updated/acknowledged/closed` lifecycle with acknowledgement persistence during later updates.

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

## Remaining external gate

- Freeze zone/rule/class policy on leakage-free event replays and report event precision/recall, duplicate rate,
  detection delay, and false alarms/hour by location, weather, and sea state.
