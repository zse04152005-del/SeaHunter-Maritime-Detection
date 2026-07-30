# M2 ReID quality and tracklet-template progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: cloud accepted; real maritime calibration remains open

## Implemented

- Replaceable `AppearanceEncoder` contract plus a lightweight normalized BGR-histogram baseline.
- Crop gates for minimum short side, area, visible fraction, overlap/occlusion, brightness, saturation, sharpness,
  payload shape, and finite pixels.
- Quality decisions occur before encoding, so rejected crops do not consume ReID inference or update memory.
- Eligible clear high-confidence observations update a normalized tracklet template through quality-weighted
  exponential aggregation; low-confidence second-stage matches cannot contaminate the template.
- A third Hungarian association layer matches unmatched high-confidence detections to recent confirmed/lost templates
  using class, relaxed Kalman motion, and minimum cosine-similarity gates.
- Track audit fields record `association_stage=reid`, accepted appearance score, eligibility, and bypass reason.
- Sequence summaries count eligible observations, each bypass reason, template updates, evaluated/gated pairs, and
  successful ReID matches.
- The CLI exposes every quality, similarity, age, template, and motion threshold; ReID is valid only with BoT-SORT.

## Tiny-target policy

Targets below the configured short-side or area threshold are labeled `target_too_small`. They never invoke the
appearance encoder, never update a template, and cannot reacquire a lost identity through ReID. Their association
continues through detector confidence, Kalman/GMC motion, and later M3 world-coordinate evidence.

## Acceptance boundary

The histogram encoder is an edge-friendly integration baseline, not a claim of discriminative maritime ReID quality.
Synthetic tests use scripted embeddings to prove lifecycle, gating, template, and audit behavior. The deployment
configuration stays disabled until a held-out real maritime crop/track set calibrates gates and compares a validated
ONNX encoder. No real ID-switch or reacquisition-rate improvement is claimed here.

## Cloud acceptance

- Accepted commit: `50122476b86265673924ea396101b170d754ed12`.
- The focused `m2-reid` job passed 19 tests in 1.34 seconds.
- Cloud validation:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30536083275>
- Python 3.10/3.12 CI and quality gates:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30536083108>
- ReID artifact:
  <https://github.com/zse04152005-del/SeaHunter-Maritime-Detection/actions/runs/30536083275/artifacts/8756699258>

## Remaining external gate

- Calibrate crop gates, similarity thresholds, and the final encoder on leakage-free held-out maritime tracks before
  enabling ReID in a deployment configuration.
