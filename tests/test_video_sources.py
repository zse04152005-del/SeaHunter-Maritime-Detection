from __future__ import annotations

import unittest

from seahunter.schemas import SourceKind
from seahunter.video import parse_video_source


class VideoSourceTests(unittest.TestCase):
    def test_rtsp_is_not_treated_as_file(self) -> None:
        source = parse_video_source("rtsp://camera.example/live")
        self.assertEqual(source.kind, SourceKind.RTSP)
        self.assertTrue(source.is_network)
        self.assertTrue(source.is_live)

    def test_srt_source(self) -> None:
        source = parse_video_source("srt://127.0.0.1:9000?mode=listener")
        self.assertEqual(source.kind, SourceKind.SRT)

    def test_numeric_source_is_device(self) -> None:
        source = parse_video_source("0")
        self.assertEqual(source.kind, SourceKind.DEVICE)

    def test_local_path_does_not_need_to_exist_during_parsing(self) -> None:
        source = parse_video_source("samples/flight.mp4")
        self.assertEqual(source.kind, SourceKind.FILE)

    def test_unknown_scheme_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_video_source("ftp://example.com/video.mp4")

    def test_suggested_network_id_does_not_expose_credentials(self) -> None:
        source = parse_video_source("rtsp://operator:secret@camera.example/live")
        suggested = source.suggested_id()
        self.assertNotIn("operator", suggested)
        self.assertNotIn("secret", suggested)
        self.assertTrue(suggested.startswith("rtsp-camera-example-"))


if __name__ == "__main__":
    unittest.main()
