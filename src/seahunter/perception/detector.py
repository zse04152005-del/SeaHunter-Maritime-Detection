"""Detector protocol implemented by PyTorch, ONNX, or TensorRT backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from seahunter.schemas import Detection, FramePacket


class Detector(Protocol):
    """Minimal detector contract used by the video pipeline."""

    @property
    def detector_id(self) -> str:
        """Return a stable model/backend identifier."""

    def infer(self, frame: FramePacket) -> Sequence[Detection]:
        """Run inference for one frame and return normalized detections."""
