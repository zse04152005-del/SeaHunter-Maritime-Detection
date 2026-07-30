# M2 telemetry motion-prior progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; real synchronized telemetry benefit gate remains open

## Implemented boundary

- Bounded, timestamp-ordered `TelemetryPacket` history per source with deterministic duplicate handling.
- Nearest-sample alignment gate, telemetry quality gate, wrapped angle deltas, and implausible-rotation rejection.
- Platform plus gimbal roll/pitch/yaw converted into a small-angle image affine using an explicitly configured focal
  length.
- Quality-weighted visual/telemetry affine fusion when estimates agree, with higher-quality source selection when
  they disagree.
- Visual rejection can fall back to telemetry; telemetry rejection preserves the visual GMC result.
- `FrameResult.telemetry` and `TrackingSink` pass aligned packets to telemetry-aware trackers without changing the
  generic tracker update signature.
- Track audit JSONL schema 3 records selected source, visual/prior qualities, fusion reason, affine, and fallback.
- Sequence summaries count applied motion by `visual`, `telemetry`, or `fused` source.

## Safety boundary

The prior models rotational camera motion only. It does not use GPS translation, estimate target world position, or
claim absolute range. The default deployment config keeps it disabled until focal length and video/telemetry timing
are calibrated. Missing, stale, repeated, low-quality, non-finite, or implausibly large telemetry is rejected rather
than silently converted into motion.

Synthetic tests validate coordinate signs, yaw wrap, time gates, fusion branches, audit fields, and tracker identity
continuity for a known camera rotation. They are implementation checks only and set no real maritime performance
claim. A real benefit gate requires synchronized target-platform telemetry and held-out moving-camera maritime video.

## Cloud acceptance

- Accepted commit: `288fc39fd4e315a8d3018dc67194268f751fe7d6`.
- The focused `m2-motion-prior` job passed 17 tests in 1.32 seconds.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30534512793>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30534512383>
- Motion-prior artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30534512793/artifacts/8756069632>

## Remaining external gate

- Real telemetry/video calibration and paired visual-only versus fused tracking evaluation remain external M3/M8
  gates.
