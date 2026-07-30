"""Deterministic decoder build-capability audit helpers."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecoderBuildReport:
    """Capabilities advertised by the imported OpenCV build.

    A CUDA-enabled OpenCV build or visible CUDA device does not prove NVDEC is
    used for a particular stream, so ``nvdec_verified`` is always false here.
    Per-source selection is reported separately by ``OpenCVFrameReader``.
    """

    schema_version: int
    opencv_version: str
    ffmpeg_compiled: bool | None
    gstreamer_compiled: bool | None
    cuda_compiled: bool | None
    hardware_acceleration_property_supported: bool
    hardware_device_property_supported: bool
    cuda_device_count: int | None
    nvdec_verified: bool
    verification_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report with stable field names."""

        return asdict(self)


def probe_opencv_decoder_build(cv2_module: Any | None = None) -> DecoderBuildReport:
    """Inspect one OpenCV module without opening media or claiming runtime use."""

    cv2 = cv2_module if cv2_module is not None else importlib.import_module("cv2")
    build_information = _safe_build_information(cv2)
    cuda_device_count = _safe_cuda_device_count(cv2)
    return DecoderBuildReport(
        schema_version=1,
        opencv_version=str(getattr(cv2, "__version__", "unknown")),
        ffmpeg_compiled=_build_flag(build_information, "FFMPEG"),
        gstreamer_compiled=_build_flag(build_information, "GStreamer"),
        cuda_compiled=_build_flag(build_information, "NVIDIA CUDA"),
        hardware_acceleration_property_supported=hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
        and hasattr(cv2, "VIDEO_ACCELERATION_ANY"),
        hardware_device_property_supported=hasattr(cv2, "CAP_PROP_HW_DEVICE"),
        cuda_device_count=cuda_device_count,
        nvdec_verified=False,
        verification_reason=(
            "Build inspection cannot prove per-stream NVDEC use; validate an opened source on the target NVIDIA device."
        ),
    )


def _safe_build_information(cv2: Any) -> str:
    getter = getattr(cv2, "getBuildInformation", None)
    if not callable(getter):
        return ""
    try:
        return str(getter())
    except Exception:
        return ""


def _safe_cuda_device_count(cv2: Any) -> int | None:
    cuda = getattr(cv2, "cuda", None)
    getter = getattr(cuda, "getCudaEnabledDeviceCount", None)
    if not callable(getter):
        return None
    try:
        value = int(getter())
    except Exception:
        return None
    return max(0, value)


def _build_flag(build_information: str, key: str) -> bool | None:
    expected = key.casefold()
    for raw_line in build_information.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        candidate, raw_value = line.split(":", 1)
        if candidate.strip().casefold() != expected:
            continue
        value = raw_value.strip().split(maxsplit=1)[0].casefold()
        if value in {"yes", "true", "on"}:
            return True
        if value in {"no", "false", "off"}:
            return False
        return None
    return None
