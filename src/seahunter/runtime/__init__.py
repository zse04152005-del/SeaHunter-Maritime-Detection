"""Runtime adapters for legacy, ONNX, and TensorRT backends."""

from .legacy_ultralytics import activate_legacy_ultralytics, legacy_source_path

__all__ = ["activate_legacy_ultralytics", "legacy_source_path"]
