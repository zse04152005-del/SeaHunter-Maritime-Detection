"""Audit a maritime dataset manifest for leakage and coverage gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seahunter.data import audit_dataset, load_dataset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_dataset(load_dataset_manifest(args.manifest))
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 2 if args.require_gates and not report.passes_blocking_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
