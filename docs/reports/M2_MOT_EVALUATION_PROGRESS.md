# M2 MOT evaluation progress report

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Outcome

SeaHunter-VIS now provides a standard MOTChallenge evaluation path plus a separate frame-level diagnostic index.
Standard scores are delegated to the official TrackEval implementation; SeaHunter-VIS owns strict maritime input
validation, class-agnostic normalization, report schema, and error traceability.

## Implemented

- Strict MOT text parsing with positive one-based frame/track IDs and finite positive boxes.
- Duplicate identity detection within each frame.
- Ground-truth mark, class, visibility, and prediction-confidence parsing.
- Safe sequence/tracker/benchmark identifiers that cannot escape evaluation directories.
- Class-agnostic normalization to TrackEval's MOTChallenge 2D-box adapter.
- Official HOTA, DetA, AssA, LocA, CLEAR MOT, and identity metric orchestration.
- Normalized JSON metrics for HOTA, MOTA, MOTP, recall, precision, IDF1, ID switches, fragmentation, FP, and FN.
- Deterministic normalized MOT inputs and native TrackEval summary/detailed artifacts.
- Independent Hungarian/IoU diagnostic matching for exact error-frame indexing.
- Per-frame false-negative IDs, false-positive IDs, ID-switch mappings, and fragmentation IDs.
- Synthetic maritime sequence containing one miss, one false positive, one ID switch, and one fragmentation.
- CLI entry point and source-checkout wrapper.

## Dependency decision

The PyPI TrackEval 1.1–1.3 packages require NumPy 2.3+ on Python 3.11+, conflicting with the current verified
video/tracking upper bound. SeaHunter-VIS therefore pins official MIT-licensed TrackEval commit
`12c8791b303e0a0b50f753af204249e622d0281a`, whose package metadata depends only on NumPy and SciPy. This retains
the standard metric implementation without forcing an unvalidated NumPy migration across OpenCV, Ultralytics, and
edge environments. Because that official revision still calls the removed `np.float`, `np.int`, and `np.bool`
aliases, the SeaHunter-VIS import boundary restores only those three aliases before loading TrackEval. The upstream
source remains unmodified, and Python 3.10/3.12 cloud tests cover the compatibility bridge.

## Cloud verification gate

The normal Python 3.10/3.12 matrix installs the `evaluation` extra and runs the TrackEval-backed synthetic test. The
focused `m2-evaluation` workflow additionally executes the CLI and uploads normalized metrics, the error index,
normalized inputs, and native TrackEval outputs. The ROADMAP MOT evaluation/visualization task remains open until
these jobs pass.

## Dataset acceptance still required

Synthetic tests validate evaluator wiring and known error counts, not maritime tracker quality. Publishing real
HOTA/IDF1 claims still requires a leakage-free video split, documented ignore regions, target visibility policy,
weather/camera-motion strata, and versioned annotations.
