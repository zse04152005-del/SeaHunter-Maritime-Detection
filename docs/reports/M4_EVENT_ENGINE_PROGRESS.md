# M4 danger-zone and event-engine progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

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
