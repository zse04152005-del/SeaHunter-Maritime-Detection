# M3 calibration, telemetry, and coordinate foundation

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; target-aircraft calibration remains open

## Implemented

- Strict versioned camera intrinsics/extrinsics manifest with finite, orthonormal, right-handed validation.
- Iterative OpenCV-compatible radial/tangential pixel undistortion and normalized camera rays.
- Explicit optical, gimbal/body FRD, MAVLink NED, product ENU, WGS84 ECEF, and geodetic transformations.
- Dependency-light aggregation of common `GLOBAL_POSITION_INT`, `ATTITUDE`, and `MOUNT_ORIENTATION` messages.
- Bounded affine autopilot-clock calibration with offset, drift, RMSE, quality, and reset reporting.
- Bounded telemetry history, configurable video-latency correction, nearest-frame alignment, and quality/error gates.
- Calibration audit CLI, reference configs, operating procedure, and focused cloud suite.

## Safety boundary

The example calibration is not a field calibration. The MAVLink adapter emits no packet before position, attitude,
and accepted clock evidence exist. Frame alignment returns an explicit rejected result for missing, stale, or
low-quality telemetry. This increment establishes coordinate and timing contracts only; it does not yet claim range,
world position, or target-device synchronization accuracy.

## Cloud acceptance

- Accepted commit: `6542d43c666460547afab11e3d83691edd7c4483`.
- The focused `m3-foundation` job passed 22 tests in 1.29 seconds and generated a calibration audit.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538095453>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538095442>
- M3 foundation artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538095453/artifacts/8757502911>

The first cloud attempt exposed a missing tracking extra and an unrealistically strict sub-0.01-ppm assertion relative
to microsecond UTC timestamps. The accepted rerun uses a still-conservative 0.1-ppm synthetic tolerance; deployment
gates remain those in the versioned synchronization config.

## Remaining external gate

- Capture and approve real intrinsics, extrinsics, vertical datum, clock drift, and end-to-end video latency for each
  target camera/lens/mount/autopilot/firmware profile.
