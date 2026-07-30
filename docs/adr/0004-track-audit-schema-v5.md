# ADR 0004: Track audit JSONL schema 5

Date: 2026-07-30
Status: accepted

## Context

Schema 4 distinguishes direct observations from inferred points but cannot tell whether an inferred point is a
causal Kalman extrapolation or a hindsight interpolation after the identity is observed again. Treating both as the
same evidence can contaminate evaluation, visualization, and downstream event logic.

## Decision

Schema 5 preserves every schema 4 field and adds the nullable root key `inference_method`:

- observed points must use `null`;
- causal lost-track predictions use `extrapolated`;
- points backfilled inside a bounded, subsequently closed gap use `interpolated`.

The online tracker remains causal. Interpolation is an optional output writer with a bounded frame delay and never
changes the live preview state. It retains the Kalman extrapolation covariance rather than claiming that linear
hindsight interpolation removes uncertainty. MOTChallenge output remains unchanged.

## Compatibility

Consumers that ignore unknown keys can read schema 5. Strict version checks must add version 5. A record can be
downgraded to schema 4 by removing `inference_method` and setting `schema_version` to 4; all schema 4 values retain
their meaning.
