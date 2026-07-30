# ADR 0002: Track audit JSONL schema 3

Date: 2026-07-30
Status: accepted

## Context

Schema 2 records the selected global-motion affine, quality, applied flag, and fallback reason. It cannot distinguish
visual GMC from an IMU/gimbal prior or explain a fusion decision, which makes field debugging and safety review
ambiguous.

## Decision

Track audit JSONL schema 3 preserves every schema 2 field and adds four optional keys under `global_motion`:

- `source`: `visual`, `telemetry`, `fused`, or `none`;
- `visual_quality`;
- `prior_quality`;
- `fusion_reason`.

MOTChallenge output is unchanged. Preview track records receive the same optional keys. ByteTrack records continue to
set `global_motion` to `null`.

## Compatibility

Consumers that ignore unknown JSON object keys can read schema 3 without changes. Consumers that enforce
`schema_version == 2` must explicitly allow version 3 before deployment. A schema 3 record can be downgraded to
schema 2 by deleting the four new keys and setting `schema_version` to 2; no existing value changes meaning.

## Consequences

Fusion and fallback choices become replay-auditable. Strict consumers must perform a small version migration, while
standard MOT evaluation files and existing ByteTrack behavior remain stable.
