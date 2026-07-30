#!/usr/bin/env python3
"""Run or dry-run a strict configuration-driven detector experiment matrix."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seahunter.tools.experiment_runner import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
