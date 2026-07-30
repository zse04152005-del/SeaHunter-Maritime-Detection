# Camera, mount, and telemetry calibration procedure

## Scope and coordinate conventions

- Image pixels: origin at top-left, `u` right, `v` down.
- Camera optical: `x` right, `y` down, `z` forward.
- Gimbal and aircraft body: FRD, `x` forward, `y` right, `z` down.
- Navigation: MAVLink NED; product geometry: ENU.
- Geodetic: WGS84 latitude, longitude, ellipsoidal altitude unless a mission profile explicitly supplies another
  vertical datum.

MAVLink `relative_alt` is labeled `relative_home`, while `GLOBAL_POSITION_INT.alt` is labeled `amsl`; neither may be
silently treated as height above the instantaneous sea surface.

Never reuse a calibration after changing camera, lens, focus/zoom lock, resolution/crop, mount, vibration isolator,
gimbal firmware convention, or image rotation.

## Intrinsic calibration

1. Lock resolution, crop, focus, zoom, stabilization, and all digital transforms used in flight.
2. Capture at least 20 sharp checkerboard/ChArUco views covering the center, edges, corners, scale, and tilt range.
3. Reject blurred frames and frames with incomplete board geometry; retain the raw capture set and board dimensions.
4. Fit pinhole plus `k1,k2,p1,p2,k3`; inspect per-view residuals rather than accepting only aggregate RMS.
5. Hold out several views and validate reprojection independently. Freeze the RMS threshold before acceptance.
6. Record camera serial, lens, settings, temperature range, image dimensions, timestamp, tool version, and raw-data
   hash in the calibration package.

## Mount/gimbal extrinsics

1. Measure camera optical-to-gimbal FRD axes and lever arm with the aircraft level and gimbal at its reference pose.
2. Solve rotation using surveyed targets across multiple gimbal attitudes; verify determinant `+1` and orthogonality.
3. Verify signs with three tests: forward optical ray, positive yaw, and downward pitch.
4. Validate against held-out known bearings/ranges. Do not tune extrinsics on the final geolocation test set.

## Clock and latency calibration

1. Timestamp frames at capture/PTS when available and record decode arrival separately.
2. Pair autopilot boot time with UTC/reference arrival samples over the full mission duration.
3. Fit offset and scale; reject excessive residual or drift and reset on a boot-clock rollback.
4. Measure camera exposure, encoder, transport, buffering, and decoder latency with a common visible/electrical event.
5. Set `video_latency_seconds`, then validate nearest telemetry alignment across start, middle, and end of a flight.
6. Monitor drift, RMSE, alignment error, clock resets, and sample count in operation. Quality-gate absolute geometry.

## Audit commands

```powershell
seahunter-calibration-audit `
  configs/geometry/camera.example.json `
  --output reports/camera-calibration-audit.json
```

The example file is a schema/reference only and must never be used as a real aircraft calibration.
