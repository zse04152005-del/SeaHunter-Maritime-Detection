from __future__ import annotations

import unittest
from unittest.mock import patch

from seahunter.video import OpenCVReaderConfig, create_opencv_capture, parse_video_source, probe_opencv_decoder_build


class FakeCuda:
    @staticmethod
    def getCudaEnabledDeviceCount() -> int:
        return 2


class FakeCapture:
    def __init__(self, *, opened: bool, acceleration: int = 0, backend: str = "FFMPEG") -> None:
        self.opened = opened
        self.acceleration = acceleration
        self.backend = backend
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def getBackendName(self) -> str:
        return self.backend

    def get(self, property_id: int) -> float:
        if property_id == FakeCv2.CAP_PROP_HW_ACCELERATION:
            return float(self.acceleration)
        if property_id == FakeCv2.CAP_PROP_HW_DEVICE:
            return -1.0
        return 0.0

    def release(self) -> None:
        self.released = True


class FakeCv2:
    __version__ = "4.99.0-test"
    CAP_ANY = 0
    CAP_FFMPEG = 1900
    CAP_GSTREAMER = 1800
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    CAP_PROP_HW_ACCELERATION = 50
    CAP_PROP_HW_DEVICE = 51
    VIDEO_ACCELERATION_NONE = 0
    VIDEO_ACCELERATION_ANY = 1
    VIDEO_ACCELERATION_VAAPI = 3
    cuda = FakeCuda()

    def __init__(self, captures: list[FakeCapture]) -> None:
        self.captures = list(captures)
        self.calls: list[tuple[object, ...]] = []

    @staticmethod
    def getBuildInformation() -> str:
        return """
          Video I/O:
            FFMPEG: YES
            GStreamer: NO
          NVIDIA CUDA: YES (ver 12.8)
        """

    def VideoCapture(self, *args: object) -> FakeCapture:
        self.calls.append(args)
        return self.captures.pop(0)


class DecoderCapabilityTests(unittest.TestCase):
    def test_build_report_distinguishes_compiled_backends_from_nvdec_proof(self) -> None:
        report = probe_opencv_decoder_build(FakeCv2([]))

        self.assertEqual(report.opencv_version, "4.99.0-test")
        self.assertTrue(report.ffmpeg_compiled)
        self.assertFalse(report.gstreamer_compiled)
        self.assertTrue(report.cuda_compiled)
        self.assertEqual(report.cuda_device_count, 2)
        self.assertTrue(report.hardware_acceleration_property_supported)
        self.assertFalse(report.nvdec_verified)

    def test_failed_hardware_open_retries_without_acceleration_and_is_audited(self) -> None:
        hardware_capture = FakeCapture(opened=False)
        software_capture = FakeCapture(opened=True)
        cv2 = FakeCv2([hardware_capture, software_capture])

        with patch("seahunter.video.reader.importlib.import_module", return_value=cv2):
            capture = create_opencv_capture(
                parse_video_source("rtsp://camera.example/live"),
                OpenCVReaderConfig(backend="ffmpeg", prefer_hardware_decode=True),
            )

        selection = capture.decoder_selection()  # type: ignore[attr-defined]
        self.assertEqual(len(cv2.calls), 2)
        self.assertEqual(cv2.calls[0][-1][-2:], [FakeCv2.CAP_PROP_HW_ACCELERATION, FakeCv2.VIDEO_ACCELERATION_ANY])
        self.assertNotIn(FakeCv2.CAP_PROP_HW_ACCELERATION, cv2.calls[1][-1])
        self.assertTrue(hardware_capture.released)
        self.assertTrue(selection.hardware_attempted)
        self.assertTrue(selection.hardware_fallback)
        self.assertIn("retried without acceleration", selection.fallback_reason or "")
        self.assertEqual(selection.selected_backend, "FFMPEG")
        self.assertEqual(selection.effective_hardware_acceleration, "none")
        self.assertFalse(selection.nvdec_verified)

    def test_report_records_effective_acceleration_without_claiming_nvdec(self) -> None:
        cv2 = FakeCv2([FakeCapture(opened=True, acceleration=FakeCv2.VIDEO_ACCELERATION_VAAPI)])

        with patch("seahunter.video.reader.importlib.import_module", return_value=cv2):
            capture = create_opencv_capture(
                parse_video_source("rtsp://camera.example/live"),
                OpenCVReaderConfig(backend="ffmpeg", prefer_hardware_decode=True),
            )

        selection = capture.decoder_selection()  # type: ignore[attr-defined]
        self.assertFalse(selection.hardware_fallback)
        self.assertEqual(selection.effective_hardware_acceleration, "vaapi")
        self.assertFalse(selection.nvdec_verified)


if __name__ == "__main__":
    unittest.main()
