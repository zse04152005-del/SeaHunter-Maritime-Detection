# M2 BoT-SORT and visual GMC progress report

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: accepted by cloud validation

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

## Cloud verification

Commit `3966591` passed both required GitHub Actions workflows:

- [CI run 30529822102](https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30529822102):
  quality checks and the complete unit/integration suite passed on Python 3.10 and 3.12.
- [Cloud validation run 30529821909](https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30529821909):
  all 13 focused `m2-gmc` tests passed, the synthetic camera-pan CLI experiment completed, and its JUnit and JSON
  reports were uploaded. M1 replay, M2 tracking, M2 evaluation, and the original-weight CPU regression also passed.

All three estimable frame transitions accepted GMC, with mean RANSAC inlier quality `0.984727` and maximum
translation error `0.019081 px`. With otherwise identical tracking settings, the deterministic four-frame pan used
track IDs `[1, 2, 3, 4]` and produced three ID switches without GMC; visual GMC retained track ID `[1]` and produced
zero ID switches. The report explicitly records `dataset_claim: false`.

## Real-data acceptance still required

The implementation task is accepted, but the ROADMAP real moving-camera benefit gate remains open until the same
detector outputs and tracker parameters are evaluated with GMC disabled/enabled on a leakage-free maritime benchmark
stratified by camera motion, target size,
sea state, visibility, and weather. Reports must include HOTA, AssA, IDF1, ID switches, fragmentation, GMC fallback
rate, and per-frame failure review.
