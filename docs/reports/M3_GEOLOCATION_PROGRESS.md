# M3 sea-plane geolocation, filtering, trend, and TTC progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Implemented

- Bottom-center target contact ray through distortion, camera mount, gimbal, body, NED, and ENU transforms.
- Local sea-plane intersection with strict resolution, altitude-datum, horizon, positive-height, and range gates.
- Altitude, tide, wave, attitude, and reprojection uncertainty propagation into range and horizontal covariance.
- Absolute latitude/longitude only above a configurable combined calibration/telemetry/alignment/uncertainty quality.
- Per-track constant-velocity ENU EKF with process noise, Joseph covariance update, monotonic time, and gap reset.
- Relative radial/tangential motion classification and quality-gated TTC with explicit degradation reasons.

## Safety and acceptance boundary

The sea is locally approximated as horizontal. AMSL, relative-home, and WGS84 ellipsoid heights are never mixed.
Near-horizon rays and low-quality solutions do not emit absolute coordinates. Linear synthetic geometry validates
signs, gates, covariance plumbing, filtering, and TTC math only; accuracy claims require truth from the actual
aircraft, camera, vertical datum, sea state, and mission envelope.
