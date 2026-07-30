# M3 sea-plane geolocation, filtering, trend, and TTC progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; real truth-set accuracy gate remains open

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

## Cloud acceptance

- Accepted commit: `7fef086c8a113dc7ac022b43ef2077deb12f94d7`.
- The focused `m3-geolocation` job passed 21 tests in 0.28 seconds.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538754473>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538754478>
- M3 geolocation artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30538754473/artifacts/8757776017>

## Remaining external gate

- Build the versioned RTK/AIS/rangefinder/known-marker truth set and freeze distance/attitude/sea-state MAE and
  relative-error requirements before claiming operational range, geolocation, trend, or TTC accuracy.
