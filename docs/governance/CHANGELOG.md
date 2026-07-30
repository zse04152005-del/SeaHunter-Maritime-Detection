# Engineering change log

## 2026-07-30

- Froze the original detector, custom source, weight hash, framework version, and licensing boundary.
- Restored the incomplete local Ultralytics data package and made local-source activation deterministic.
- Connected configurable CIoU/NWD loss blending and validation.
- Reorganized historical code and paper assets under `legacy/` and `docs/archive/`.
- Added locked packaging, Ruff, strict mypy, GitHub Actions, legacy-weight CPU smoke testing, and ONNX consistency.
- Added bounded video ingestion, replay, preview, Parquet/JSONL outputs, monitoring, and reconnect behavior.
- Added ByteTrack, track audit/MOT output, visualization, TrackEval evaluation, BoT-SORT visual GMC, and cloud ablations.
- Replaced the legacy single-entry pseudo-ablation with a strict multi-variant, three-seed experiment plan and dry-run
  validation path.
- Added auditable OpenCV decoder build/selection reports and deterministic hardware-to-software fallback coverage;
  target-device NVDEC acceptance remains explicitly open.

This log summarizes accepted engineering increments. Exact implementation and validation evidence remains in Git
history and `docs/reports/`.
