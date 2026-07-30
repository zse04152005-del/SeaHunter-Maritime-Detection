# M3 calibration, telemetry, and coordinate foundation

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

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
