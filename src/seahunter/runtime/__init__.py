"""Runtime adapters for legacy, ONNX, and TensorRT backends."""

from .legacy_ultralytics import activate_legacy_ultralytics, legacy_source_path
from .ultralytics_detector import UltralyticsDetector

__all__ = ["UltralyticsDetector", "activate_legacy_ultralytics", "legacy_source_path"]
