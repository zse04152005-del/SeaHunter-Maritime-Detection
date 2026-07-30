# M2 BoT-SORT and visual GMC progress report

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Outcome

SeaHunter-VIS now has a selectable BoT-SORT motion baseline for moving-camera video. It estimates a
previous-to-current-frame affine warp from background features and applies accepted transforms to Kalman means and
covariance before association. Low-quality or implausible estimates fall back to identity and remain visible in audit
output.

This increment deliberately keeps ReID disabled. Maritime UAV targets may be too small, blurred, occluded, or
degraded for reliable appearance embeddings; the later ReID task must first define target-size and image-quality
gates plus clear-frame tracklet templates.

## Implemented

- `bytetrack` / `botsort` replay selection behind the existing tracker protocol.
- Sparse Shi-Tomasi background features and pyramidal Lucas-Kanade optical flow.
- Previous/current detector-box exclusion with configurable padding.
- RANSAC partial-affine fitting and inlier count/ratio gates.
- Translation, scale, and rotation plausibility rejection.
- Identity fallback for first frame, missing/invalid payload, insufficient features/flow/inliers, failed fitting,
  and implausible transforms.
- Affine propagation of Kalman center, box size, velocity, and covariance.
- Stable tracker/configuration identifiers with explicit `reid=0`.
- Per-frame affine, quality, applied flag, and fallback reason in track audit JSONL schema v2.
- Sequence-level GMC application-rate and mean-quality summary in replay output.
- Versioned starting configuration in `configs/tracking/botsort.maritime.yaml`.
- Deterministic synthetic camera-pan no-GMC/GMC ablation and focused cloud job.

## Cloud verification gate

The Python 3.10/3.12 CI matrix must pass the complete test suite. The focused `m2-gmc` job additionally runs the
GMC estimator, BoT-SORT integration, audit-output, CLI, and synthetic ablation tests, generates a JSON comparison,
and uploads the JUnit and ablation artifacts.

Expected synthetic behavior is three no-GMC identity switches versus zero with GMC over four translated frames.
This proves deterministic estimator/tracker integration only; it is not a real maritime dataset claim.

## Real-data acceptance still required

The ROADMAP moving-camera benefit gate remains open until the same detector outputs and tracker parameters are
evaluated with GMC disabled/enabled on a leakage-free maritime benchmark stratified by camera motion, target size,
sea state, visibility, and weather. Reports must include HOTA, AssA, IDF1, ID switches, fragmentation, GMC fallback
rate, and per-frame failure review.
