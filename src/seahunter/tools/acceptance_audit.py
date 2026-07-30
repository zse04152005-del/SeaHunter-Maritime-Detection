"""Audit a sea-trial submission without granting or fabricating approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seahunter.acceptance import audit_acceptance, load_acceptance_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-passed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan, trials, reviews = load_acceptance_submission(args.submission)
    audit = audit_acceptance(plan, trials, reviews)
    rendered = json.dumps(audit.to_dict(), sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 2 if args.require_passed and not audit.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
