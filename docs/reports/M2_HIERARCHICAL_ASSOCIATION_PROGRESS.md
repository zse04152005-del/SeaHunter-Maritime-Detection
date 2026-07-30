# M2 hierarchical association progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Implemented

- Preserved ByteTrack's high-confidence first stage and confirmed-track low-confidence second stage.
- Added a configurable squared Mahalanobis gate in `(cx, cy, width, height)` measurement space before Hungarian
  assignment; the default `13.276704` is the chi-square 99% quantile for four dimensions.
- Invalid motion or below-threshold IoU pairs receive an explicit prohibitive cost before assignment instead of being
  assigned and rejected afterward.
- Added sequence counters for matches by stage, evaluated motion pairs, gated pairs, and gate rate.
- Track audit JSONL schema 4 records association stage and accepted motion distance. Appearance/ReID audit keys are
  reserved as nullable fields for the next M2 increment.
- Replay summaries expose association diagnostics for ByteTrack and BoT-SORT.

## Acceptance boundary

Synthetic tests exercise high/low association, a deliberately implausible scale jump with non-zero IoU, Kalman
distance ordering, audit output, and counter reset. These tests prove deterministic gating behavior only. Threshold
calibration and ID-switch benefit require the held-out maritime MOT corpus and remain an external data gate.
