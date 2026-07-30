"""CLI for auditable OpenCV decode build and per-source selection reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from seahunter.video import OpenCVFrameReader, OpenCVReaderConfig, parse_video_source, probe_opencv_decoder_build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="optional file/device/RTSP/SRT/HTTP source to open and decode once")
    parser.add_argument("--backend", choices=("auto", "ffmpeg", "gstreamer"), default="ffmpeg")
    parser.add_argument("--software-decode", action="store_true", help="do not request hardware acceleration")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path instead of stdout")
    return parser


def create_report(
    *, source_value: str | None, backend: Literal["auto", "ffmpeg", "gstreamer"], software_decode: bool
) -> dict[str, object]:
    """Build one stable report; source credentials and locations are never serialized."""

    report: dict[str, object] = {
        "schema_version": 1,
        "build": probe_opencv_decoder_build().to_dict(),
        "source_probe": None,
    }
    if source_value is None:
        return report

    source = parse_video_source(source_value)
    reader = OpenCVFrameReader(
        source,
        config=OpenCVReaderConfig(
            backend=backend,
            prefer_hardware_decode=not software_decode,
            max_reconnect_attempts=0,
        ),
    )
    probe: dict[str, object] = {
        "source_kind": source.kind.value,
        "opened": False,
        "first_frame_decoded": False,
        "selection": None,
        "error_type": None,
        "error": None,
    }
    try:
        frame = reader.read()
        probe["opened"] = reader.stats().open_count > 0
        probe["first_frame_decoded"] = frame is not None
    except Exception as exc:
        probe["opened"] = reader.stats().open_count > 0
        probe["error_type"] = type(exc).__name__
        probe["error"] = str(exc)
    finally:
        selection = reader.decoder_selection()
        if selection is not None:
            probe["selection"] = asdict(selection)
        reader.close()
    report["source_probe"] = probe
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = create_report(
        source_value=args.source,
        backend=args.backend,
        software_decode=args.software_decode,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    source_probe = payload["source_probe"]
    return 1 if isinstance(source_probe, dict) and source_probe["error_type"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
