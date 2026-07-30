# M2 ByteTrack baseline progress report

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Outcome

SeaHunter-VIS now has a framework-neutral ByteTrack baseline connected to the video replay pipeline. It preserves
the distinction between detector observations and finite motion-model predictions so that trajectory recovery can
be evaluated and audited without presenting inferred points as direct evidence.

## Implemented

- Two-stage IoU association: high-confidence detections first, then low-confidence continuation.
- SciPy Hungarian assignment with optional class-aware matching.
- Constant-velocity Kalman filter over center XYWH state.
- Stable per-source identities with strict frame ordering and explicit reset semantics.
- `tentative`, `confirmed`, `lost`, and `removed` lifecycle states.
- Configurable confirmation hits, loss buffer, FPS, and inferred-confidence decay.
- High-confidence-only track creation; low-confidence detections cannot create identities.
- Optional finite lost-track prediction with `inferred` observation type and a required loss reason.
- Recovery of a lost identity when a compatible high-confidence detection reappears before expiry.
- Framework-neutral tracker and track-output sink protocols.
- MOTChallenge ten-column output, observed-only by default.
- Auditable track JSONL with lifecycle, association score, covariance, velocity, age, loss reason, and tracker ID.
- Replay CLI integration and output-path collision protection.
- Versioned starting configuration in `configs/tracking/bytetrack.maritime.yaml`.

## Cloud verification plan

The feature branch is configured for the normal GitHub Actions quality and Python 3.10/3.12 test matrix. The manual
`cloud-validation` workflow also provides a focused `m2-tracking` suite covering:

- identity continuity and high/low-confidence association;
- short occlusion, expiration, and same-ID recovery;
- class-aware separation and tentative-track confirmation;
- source/frame-order invariants;
- MOT and audit JSONL behavior, including inferred-state filtering;
- track schema invariants and replay CLI argument validation.

The ROADMAP ByteTrack item must remain unchecked until both the normal CI matrix and focused M2 cloud suite pass.

## Deliberate limitations of this baseline

- Association uses IoU and a constant-velocity image-plane model only.
- No global camera-motion compensation is included yet.
- No ReID embedding or tracklet appearance template is included yet.
- Pixel velocity uses configured source FPS; telemetry/world-coordinate motion is deferred to M3.
- Annotated video still renders detector boxes rather than track IDs and inferred trajectories.
- Thresholds have not been calibrated on a leakage-free real maritime MOT dataset.

## Next M2 increment after cloud acceptance

1. Render track IDs, lifecycle, trails, and observed/inferred styling in video and WebSocket preview.
2. Add evaluator adapters for HOTA, IDF1, AssA, ID switches, and fragmentation.
3. Establish a camera-motion baseline and then add visual GMC.
4. Add ReID quality gating and clear-frame tracklet templates for sufficiently large targets.
5. Measure benefits separately for tiny targets, moving cameras, occlusion duration, and degraded weather.
