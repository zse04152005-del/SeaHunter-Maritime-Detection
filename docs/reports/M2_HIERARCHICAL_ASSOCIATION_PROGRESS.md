# M2 hierarchical association progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: high/low/motion/ReID layers cloud accepted; real maritime calibration remains open

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

## Cloud acceptance

- Accepted commit: `70fa011e3e06bf8ecc80cc55d97c52ef97443cbc`.
- The focused `m2-association` job passed 21 tests in 1.25 seconds.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30535088756>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30535088721>
- Association artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30535088756/artifacts/8756297720>

The separately quality-gated ReID layer was accepted in commit
`50122476b86265673924ea396101b170d754ed12` by the 19-test `m2-reid` job in
<https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30536083275>. The composite ROADMAP
association item is therefore complete at the implementation/wiring level. No maritime ID-switch improvement is
claimed from the synthetic motion-gate or ReID cases; real calibration and benefit measurement remain external data
gates.
