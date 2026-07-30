#!/usr/bin/env python3
"""Generate a deterministic synthetic no-GMC versus GMC comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seahunter.evaluation.gmc_ablation import run_synthetic_gmc_ablation  # noqa: E402, I001
from seahunter.evaluation.mot import write_json  # noqa: E402, I001


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_synthetic_gmc_ablation()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
