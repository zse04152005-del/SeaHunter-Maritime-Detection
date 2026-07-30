# ADR 0003: Track audit JSONL schema 4

Date: 2026-07-30
Status: accepted

## Context

Schema 3 explains global-motion fusion but cannot show which association layer recovered a track or whether a motion
gate, appearance score, or ReID quality decision was involved. This prevents frame-level diagnosis of identity
switches and false reacquisition.

## Decision

Schema 4 preserves all schema 3 fields and adds these optional root keys:

- `association_stage`: currently `new`, `high`, or `low`; later M2 work may emit `reid`;
- `motion_gate_distance`: squared four-dimensional Mahalanobis distance;
- `appearance_score`;
- `reid_eligible`;
- `reid_bypass_reason`.

The appearance/ReID keys are introduced now and remain `null` until the independently gated ReID increment is
enabled. This avoids another contract change when that branch is added. MOTChallenge output remains unchanged.

## Compatibility

Consumers that ignore unknown keys can read schema 4. Strict version checks must add version 4. Downgrading to schema
3 requires deleting the five new keys and setting `schema_version` to 3; all schema 3 values retain their meaning.
