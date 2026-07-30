# M0 governance completion progress

Date: 2026-07-30
Branch: `feature/m2-bytetrack-baseline`
Status: implementation complete; cloud validation pending

## Reconciled evidence

The original Git history, upgrade branch, source provenance, weight hash, legacy source, archive migration, generated
file cleanup, Ultralytics pin, AGPL boundary, local package activation, configurable NWD loss, project packaging,
lockfile, formatting/type checks, GitHub Actions, custom-module forward/backward/export coverage, and original-weight
CPU smoke test were already present but not reflected in the ROADMAP checkboxes.

## Completed in this increment

- Replaced the legacy one-entry `run_ablation.py` workflow with a strict JSON experiment suite.
- Defined paired CIoU-only and 50/50 CIoU/NWD variants across seeds 0, 1, and 2.
- Added exact-key validation, unique run IDs, loss-range checks, repository path containment, and source SHA-256.
- Added a dataset/GPU-free dry-run that prints the fully resolved six-run matrix.
- Added an execution path that selects the repository-local Ultralytics fork and writes a per-run manifest containing
  Git commit, suite hash, Python/framework version, device, seed, and resolved training controls.
- Exposed NWD weight and normalization controls on the single-run training CLI.
- Added explicit risk, decision, and engineering change logs under `docs/governance/`.
- Added a focused `m0-governance` GitHub Actions job and uploaded plan/JUnit artifacts.

## Acceptance boundary

Cloud validation must prove the static baseline audit, experiment schema tests, and dry-run plan on a clean checkout.
Actual six-run training remains data/GPU-gated and cannot be claimed from plan validation. TensorRT and NVIDIA target
validation belong to M7 and remain external-hardware gates.
