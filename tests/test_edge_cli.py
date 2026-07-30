from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from seahunter.schemas import SourceKind
from seahunter.tools.edge_service import main
from seahunter.tools.video_replay import _validate_output_paths, build_parser


class EdgeCliTests(unittest.TestCase):
    def test_video_replay_parser_accepts_new_outputs(self) -> None:
        args = build_parser().parse_args(
            [
                "flight.mp4",
                "--output",
                "flight.jsonl",
                "--parquet",
                "flight.parquet",
                "--annotated-video",
                "flight.avi",
                "--preview",
                "--tracker",
                "bytetrack",
                "--tracks-jsonl",
                "tracks.jsonl",
                "--track-frame-rate",
                "25",
                "--track-trail-length",
                "24",
                "--track-emit-lost",
                "--mot-output",
                "tracks.txt",
                "--mot-include-inferred",
            ]
        )
        self.assertTrue(args.preview)
        self.assertEqual(str(args.parquet), "flight.parquet")
        self.assertEqual(args.tracker, "bytetrack")
        self.assertEqual(args.track_frame_rate, 25.0)
        self.assertEqual(args.track_trail_length, 24)
        self.assertTrue(args.track_emit_lost)
        self.assertTrue(args.mot_include_inferred)

    def test_edge_service_injects_always_on_defaults(self) -> None:
        with patch("seahunter.tools.edge_service.replay_main", return_value=0) as replay:
            self.assertEqual(main(["0", "--output", "live.jsonl"]), 0)
        forwarded = replay.call_args.args[0]
        self.assertEqual(forwarded[:4], ["--realtime", "--preview", "--max-reconnect-attempts", "-1"])

    def test_output_paths_cannot_conflict_or_overwrite_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flight.mp4"
            source.touch()
            with self.assertRaises(SystemExit):
                _validate_output_paths(
                    SourceKind.FILE,
                    str(source),
                    {"JSONL output": source},
                )
            with self.assertRaises(SystemExit):
                _validate_output_paths(
                    SourceKind.FILE,
                    str(source),
                    {
                        "JSONL output": root / "result.data",
                        "annotated video": root / "result.data",
                    },
                )


if __name__ == "__main__":
    unittest.main()
