# Legacy baseline

This directory preserves the original project entry points for audit and comparison. New development must not add features here.

- `run_ablation.py`: original single-experiment training script
- `test_seahunter.py`: original image/video inference script
- `requirements-legacy.txt`: original unpinned environment list

Use `training/` and `evaluation/` for new entry points. Legacy files may be removed only after the corresponding replacement has passed the M0 acceptance gate.
